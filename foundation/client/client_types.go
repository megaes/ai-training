package client

import (
	"strconv"
	"strings"
	"time"
)

type D map[string]any

// =============================================================================

type Error struct {
	Message string `json:"error"`
}

func (err *Error) Error() string {
	return err.Message
}

// =============================================================================

type Time struct {
	time.Time
}

func ToTime(sec int64) Time {
	return Time{
		Time: time.Unix(sec, 0),
	}
}

func (t *Time) UnmarshalJSON(data []byte) error {
	d := strings.Trim(string(data), "\"")

	num, err := strconv.Atoi(d)
	if err != nil {
		return err
	}

	t.Time = time.Unix(int64(num), 0)

	return nil
}

func (t Time) MarshalJSON() ([]byte, error) {
	data := strconv.Itoa(int(t.Unix()))
	return []byte(data), nil
}

// =============================================================================

type Function struct {
	Name      string            `json:"name"`
	Arguments map[string]string `json:"arguments"`
}

type ToolCall struct {
	Function Function `json:"function"`
}

type ChatMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

type Chat struct {
	Model   string      `json:"model"`
	Created Time        `json:"created"`
	Message ChatMessage `json:"message"`
	Done    bool        `json:"done"`
}
