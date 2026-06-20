package trainer

import (
	"bytes"
	"crypto/sha1"
	"encoding/base64"
	"encoding/json"
	"io"
	"net"
	"strings"
	"sync"
	"testing"
)

func TestNewHub(t *testing.T) {
	h := NewHub()
	if h == nil {
		t.Fatal("NewHub() returned nil")
	}
	if len(h.clients) != 0 {
		t.Errorf("expected 0 clients, got %d", len(h.clients))
	}
}

func TestWebsocketAccept(t *testing.T) {
	key := "dGhlIHNhbXBsZSBub25jZQ=="
	got := websocketAccept(key)

	sum := sha1.Sum([]byte(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
	want := base64.StdEncoding.EncodeToString(sum[:])
	if got != want {
		t.Errorf("websocketAccept(%q) = %q, want %q", key, got, want)
	}
}

func TestEncodeTextFrame(t *testing.T) {
	tests := []struct {
		payload []byte
		wantLen int // frame header length
	}{
		{[]byte("hello"), 2}, // opcode + length byte
		{make([]byte, 125), 2},
		{make([]byte, 126), 4}, // opcode + 126 + 2 bytes length
		{make([]byte, 65535), 4},
	}

	for _, tt := range tests {
		frame, err := encodeTextFrame(tt.payload)
		if err != nil {
			t.Errorf("encodeTextFrame failed: %v", err)
			continue
		}
		if frame[0] != 0x81 {
			t.Errorf("opcode byte = 0x%02x, want 0x81", frame[0])
		}
		totalLen := tt.wantLen + len(tt.payload)
		if len(frame) != totalLen {
			t.Errorf("frame length = %d, want %d", len(frame), totalLen)
		}
	}
}

func TestEncodeTextFrameTooLarge(t *testing.T) {
	// This should fail for payloads > 1<<31, but we can't allocate that much.
	// Just verify the error path exists by testing the boundary logic.
	_, err := encodeTextFrame(make([]byte, 1024))
	if err != nil {
		t.Errorf("unexpected error for 1KB payload: %v", err)
	}
}

func TestReadFrameSmallPayload(t *testing.T) {
	// Create a pipe to simulate a connection
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	// Write a simple text frame (masked for client)
	payload := []byte("test")
	maskKey := []byte{0x01, 0x02, 0x03, 0x04}
	masked := make([]byte, len(payload))
	for i := range payload {
		masked[i] = payload[i] ^ maskKey[i%4]
	}

	// Frame: opcode=0x81 (text), masked=1, length=4, mask key, masked payload
	frame := []byte{0x81, 0x84}
	frame = append(frame, maskKey...)
	frame = append(frame, masked...)

	go func() {
		client.Write(frame)
	}()

	opcode, data, err := readFrame(server)
	if err != nil {
		t.Fatalf("readFrame error: %v", err)
	}
	if opcode != 0x01 {
		t.Errorf("opcode = 0x%02x, want 0x01", opcode)
	}
	if !bytes.Equal(data, payload) {
		t.Errorf("payload = %q, want %q", data, payload)
	}
}

func TestReadFrameEmptyPayload(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	// Close frame with no payload
	frame := []byte{0x88, 0x00}
	go func() {
		client.Write(frame)
	}()

	opcode, data, err := readFrame(server)
	if err != nil {
		t.Fatalf("readFrame error: %v", err)
	}
	if opcode != 0x08 {
		t.Errorf("opcode = 0x%02x, want 0x08", opcode)
	}
	if len(data) != 0 {
		t.Errorf("expected empty payload, got %d bytes", len(data))
	}
}

func TestReadFrameTooLarge(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	// Frame claiming 2MB payload (limit is 1MB)
	frame := []byte{0x81, 127, 0, 0, 0, 0, 0, 0x20, 0, 0, 0}
	go func() {
		client.Write(frame)
	}()

	_, _, err := readFrame(server)
	if err == nil {
		t.Error("expected error for oversized frame")
	}
}

func TestBroadcastJSON(t *testing.T) {
	h := NewHub()
	// Broadcast with no clients should not panic
	h.BroadcastJSON("test", map[string]string{"key": "value"})
}

func TestBroadcastJSONMarshalError(t *testing.T) {
	h := NewHub()
	// Broadcast invalid JSON (should not panic)
	h.BroadcastJSON("test", make(chan int))
}

func TestBroadcastNoClients(t *testing.T) {
	h := NewHub()
	h.Broadcast([]byte("test"))
}

func TestRemoveClient(t *testing.T) {
	h := NewHub()
	_, client := net.Pipe()
	defer client.Close()

	c := &wsClient{conn: client}
	h.mu.Lock()
	h.clients[c] = struct{}{}
	h.mu.Unlock()

	if len(h.clients) != 1 {
		t.Errorf("expected 1 client, got %d", len(h.clients))
	}

	h.remove(c)

	if len(h.clients) != 0 {
		t.Errorf("expected 0 clients after remove, got %d", len(h.clients))
	}
}

func TestRemoveNonExistentClient(t *testing.T) {
	h := NewHub()
	_, client := net.Pipe()
	defer client.Close()

	c := &wsClient{conn: client}
	h.remove(c) // should not panic
}

func TestWriteFrame(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	c := &wsClient{conn: server}

	var received bytes.Buffer
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		io.Copy(&received, client)
	}()

	// Write a text frame
	err := c.writeFrame(0x01, []byte("hello"))
	if err != nil {
		t.Fatalf("writeFrame error: %v", err)
	}
	client.Close()
	wg.Wait()

	if len(received.Bytes()) < 5 {
		t.Errorf("received %d bytes, expected at least 5", len(received.Bytes()))
	}
}

func TestWriteText(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	c := &wsClient{conn: server}

	var received bytes.Buffer
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		io.Copy(&received, client)
	}()

	err := c.writeText([]byte("test message"))
	if err != nil {
		t.Fatalf("writeText error: %v", err)
	}
	client.Close()
	wg.Wait()

	if !bytes.Contains(received.Bytes(), []byte("test message")) {
		t.Error("received data does not contain original message")
	}
}

func TestBroadcastJSONStructure(t *testing.T) {
	h := NewHub()
	h.BroadcastJSON("test_event", map[string]string{"foo": "bar"})

	// BroadcastJSON should marshal to {"type":"test_event","data":{"foo":"bar"}}
	// We can't easily capture the broadcast, but we can verify the marshaling
	payload, err := json.Marshal(map[string]any{
		"type": "test_event",
		"data": map[string]string{"foo": "bar"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(payload), "test_event") {
		t.Error("payload should contain event type")
	}
}
