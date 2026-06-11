package trainer

import (
	"crypto/sha1"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
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
		one := make([]byte, 1)
		for {
			if _, err := conn.Read(one); err != nil {
				return
			}
		}
	}()
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
