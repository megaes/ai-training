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

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatChoice struct {
	Index   int         `json:"index"`
	Message ChatMessage `json:"message"`
}

type Chat struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created Time         `json:"created"`
	Model   string       `json:"model"`
	Message ChatMessage  `json:"message"`
	Choices []ChatChoice `json:"choices"`
}

// =============================================================================

type ChatSSEDelta struct {
	Content string `json:"content"`
}

type ChatSSEChoice struct {
	Index        int          `json:"index"`
	Delta        ChatSSEDelta `json:"delta"`
	Text         string       `json:"generated_text"`
	Probs        float32      `json:"logprobs"`
	FinishReason string       `json:"finish_reason"`
}

type ChatSSE struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created Time            `json:"created"`
	Model   string          `json:"model"`
	Choices []ChatSSEChoice `json:"choices"`
	Error   string          `json:"error"`
}

// =============================================================================

type ChatVisionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatVisionChoice represents a choice for the vision call.
type ChatVisionChoice struct {
	Index   int               `json:"index"`
	Message ChatVisionMessage `json:"message"`
}

// ChatVision represents the result for the vision call.
type ChatVision struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created Time               `json:"created"`
	Model   string             `json:"model"`
	Choices []ChatVisionChoice `json:"choices"`
}
