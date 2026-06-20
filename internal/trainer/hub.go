package trainer

import (
	"crypto/sha1"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Hub struct {
	mu      sync.Mutex
	clients map[*wsClient]struct{}
}

type wsClient struct {
	conn net.Conn
	mu   sync.Mutex
}

func NewHub() *Hub {
	return &Hub{clients: make(map[*wsClient]struct{})}
}

func (h *Hub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if strings.ToLower(r.Header.Get("Upgrade")) != "websocket" {
		http.Error(w, "websocket upgrade required", http.StatusBadRequest)
		return
	}
	key := r.Header.Get("Sec-WebSocket-Key")
	if key == "" {
		http.Error(w, "missing websocket key", http.StatusBadRequest)
		return
	}
	hijacker, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "hijacking unavailable", http.StatusInternalServerError)
		return
	}
	conn, buf, err := hijacker.Hijack()
	if err != nil {
		return
	}
	accept := websocketAccept(key)
	_, _ = fmt.Fprintf(buf, "HTTP/1.1 101 Switching Protocols\r\n")
	_, _ = fmt.Fprintf(buf, "Upgrade: websocket\r\n")
	_, _ = fmt.Fprintf(buf, "Connection: Upgrade\r\n")
	_, _ = fmt.Fprintf(buf, "Sec-WebSocket-Accept: %s\r\n\r\n", accept)
	if err := buf.Flush(); err != nil {
		_ = conn.Close()
		return
	}
	client := &wsClient{conn: conn}
	h.mu.Lock()
	h.clients[client] = struct{}{}
	h.mu.Unlock()

	go func() {
		defer h.remove(client)

		// Periodic ping keep-alive every 25 seconds
		pingTicker := time.NewTicker(25 * time.Second)
		defer pingTicker.Stop()
		done := make(chan struct{})
		go func() {
			for {
				select {
				case <-pingTicker.C:
					if err := client.writeFrame(0x09, nil); err != nil {
						close(done)
						return
					}
				case <-done:
					return
				}
			}
		}()

		for {
			opcode, payload, err := readFrame(conn)
			if err != nil {
				return
			}
			// Respond to client ping (0x09) with pong (0x0A)
			if opcode == 0x09 {
				_ = client.writeFrame(0x0A, payload)
			}
			// Discard other frames (pong, close, text, binary, etc.)
			_ = payload
		}
	}()
}

// readFrame reads a single WebSocket frame, returning opcode and payload.
// Client frames are masked; server frames are not.
func readFrame(conn net.Conn) (opcode byte, payload []byte, err error) {
	header := make([]byte, 2)
	if _, err = io.ReadFull(conn, header); err != nil {
		return 0, nil, err
	}

	opcode = header[0] & 0x0F
	masked := (header[1] & 0x80) != 0
	length := uint64(header[1] & 0x7F)

	switch length {
	case 126:
		ext := make([]byte, 2)
		if _, err = io.ReadFull(conn, ext); err != nil {
			return 0, nil, err
		}
		length = uint64(ext[0])<<8 | uint64(ext[1])
	case 127:
		ext := make([]byte, 8)
		if _, err = io.ReadFull(conn, ext); err != nil {
			return 0, nil, err
		}
		length = uint64(ext[0])<<56 | uint64(ext[1])<<48 | uint64(ext[2])<<40 | uint64(ext[3])<<32 |
			uint64(ext[4])<<24 | uint64(ext[5])<<16 | uint64(ext[6])<<8 | uint64(ext[7])
	}

	if length > 1<<20 {
		return 0, nil, errors.New("websocket frame too large")
	}

	payload = make([]byte, length)
	if length > 0 {
		if _, err = io.ReadFull(conn, payload); err != nil {
			return 0, nil, err
		}
	}

	if masked {
		key := make([]byte, 4)
		if _, err = io.ReadFull(conn, key); err != nil {
			return 0, nil, err
		}
		for i := range payload {
			payload[i] ^= key[i%4]
		}
	}

	return opcode, payload, nil
}

func (h *Hub) BroadcastJSON(eventType string, data any) {
	payload, err := json.Marshal(map[string]any{
		"type": eventType,
		"data": data,
	})
	if err != nil {
		return
	}
	h.Broadcast(payload)
}

func (h *Hub) Broadcast(payload []byte) {
	h.mu.Lock()
	clients := make([]*wsClient, 0, len(h.clients))
	for client := range h.clients {
		clients = append(clients, client)
	}
	h.mu.Unlock()

	for _, client := range clients {
		if err := client.writeText(payload); err != nil {
			h.remove(client)
		}
	}
}

func (h *Hub) remove(client *wsClient) {
	h.mu.Lock()
	if _, ok := h.clients[client]; ok {
		delete(h.clients, client)
		_ = client.conn.Close()
	}
	h.mu.Unlock()
}

func (c *wsClient) writeText(payload []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	frame, err := encodeTextFrame(payload)
	if err != nil {
		return err
	}
	_, err = c.conn.Write(frame)
	return err
}

// writeFrame writes a raw WebSocket frame (opcode + payload).
// Used for ping/pong control frames.
func (c *wsClient) writeFrame(opcode byte, payload []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	frame := []byte{0x80 | opcode}
	length := len(payload)
	switch {
	case length < 126:
		frame = append(frame, byte(length))
	case length <= 65535:
		frame = append(frame, 126, byte(length>>8), byte(length))
	default:
		frame = append(frame, 127, 0, 0, 0, 0, byte(length>>24), byte(length>>16), byte(length>>8), byte(length))
	}
	frame = append(frame, payload...)
	_, err := c.conn.Write(frame)
	return err
}

func websocketAccept(key string) string {
	sum := sha1.Sum([]byte(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
	return base64.StdEncoding.EncodeToString(sum[:])
}

func encodeTextFrame(payload []byte) ([]byte, error) {
	length := len(payload)
	frame := []byte{0x81}
	switch {
	case length < 126:
		frame = append(frame, byte(length))
	case length <= 65535:
		frame = append(frame, 126, byte(length>>8), byte(length))
	default:
		if length > 1<<31 {
			return nil, errors.New("websocket payload too large")
		}
		frame = append(frame, 127, 0, 0, 0, 0, byte(length>>24), byte(length>>16), byte(length>>8), byte(length))
	}
	frame = append(frame, payload...)
	return frame, nil
}
