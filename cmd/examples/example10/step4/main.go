// https://ampcode.com/how-to-build-an-agent
//
// This example shows you how add tool calling to the chat agent from step 1.
//
// # Running the example:
//
//	$ make example10-step4
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
)

const url = "http://localhost:11434/api/chat"

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	logger := func(ctx context.Context, msg string, v ...any) {
		s := fmt.Sprintf("msg: %s", msg)
		for i := 0; i < len(v); i = i + 2 {
			s = s + fmt.Sprintf(", %s: %v", v[i], v[i+1])
		}
		log.Println(s)
	}

	cln := client.NewSSE[client.Chat](logger)

	agent := NewAgent(cln, getUserMessage)

	return agent.Run(context.TODO())
}

// =============================================================================

type Tool interface {
	Name() string
	ToolDocument() client.D
	Call(ctx context.Context, arguments map[string]string) (client.D, error)
}

// =============================================================================

type Agent struct {
	client         *client.SSEClient[client.Chat]
	getUserMessage func() (string, bool)
	tools          []Tool
	toolDocuments  []client.D
}

func NewAgent(sseClient *client.SSEClient[client.Chat], getUserMessage func() (string, bool)) *Agent {
	tools := []Tool{
		NewReadFile(),
	}

	toolDocs := make([]client.D, len(tools))
	for i, tool := range tools {
		toolDocs[i] = tool.ToolDocument()
	}

	return &Agent{
		client:         sseClient,
		getUserMessage: getUserMessage,
		tools:          tools,
		toolDocuments:  toolDocs,
	}
}

func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D
	var inToolCall bool

	fmt.Println("Chat with qwen3 (use 'ctrl-c' to quit)")

	for {
		// CHECK IF WE ARE IN A TOOL CALL BEFORE ASKING FOR INPUT.
		if !inToolCall {
			fmt.Print("\u001b[94m\nYou\u001b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}

			conversation = append(conversation, client.D{
				"role":    "user",
				"content": userInput,
			})

			fmt.Print("\u001b[93m\nqwen3\u001b[0m: ")
		}

		inToolCall = false

		d := client.D{
			"model":          "qwen3:8b",
			"messages":       conversation,
			"max_tokens":     1000,
			"temperature":    0.1,
			"top_p":          0.1,
			"top_k":          50,
			"stream":         true,
			"tools":          a.toolDocuments,
			"tool_selection": "auto",
			"options":        client.D{"num_ctx": 32000},
		}

		ch := make(chan client.Chat, 100)
		if err := a.client.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			return fmt.Errorf("do: %w", err)
		}

		var chunks []string
		var thinking bool

		for resp := range ch {
			switch resp.Message.Content {
			case "<think>":
				thinking = true
				continue
			case "</think>":
				thinking = false
				continue
			}

			if !thinking {
				switch {
				case len(resp.Message.ToolCalls) > 0:
					result, err := a.callTools(ctx, resp.Message.ToolCalls)
					if err != nil {
						return fmt.Errorf("call tools: %w", err)
					}

					if len(result) > 0 {
						conversation = append(conversation, result)
						inToolCall = true
					}

				case resp.Message.Content != "":
					fmt.Print(resp.Message.Content)
					chunks = append(chunks, resp.Message.Content)
				}
			}
		}

		if len(chunks) > 0 {
			fmt.Print("\n")

			conversation = append(conversation, client.D{
				"role":    "assistant",
				"content": strings.Join(chunks, " "),
			})
		}
	}

	return nil
}

func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) (client.D, error) {
	for _, toolCall := range toolCalls {
		for _, tool := range a.tools {
			if toolCall.Function.Name == tool.Name() {
				resp, err := tool.Call(ctx, toolCall.Function.Arguments)
				if err != nil {
					return client.D{}, fmt.Errorf("call: %w", err)
				}
				return resp, nil
			}
		}
	}

	return client.D{}, nil
}

// =============================================================================

type ReadFileInput struct {
	Path string `json:"path"`
}

type ReadFile struct {
	name string
}

func NewReadFile() ReadFile {
	return ReadFile{
		name: "read_file",
	}
}

func (rf ReadFile) Name() string {
	return rf.name
}

func (rf ReadFile) ToolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        rf.Name(),
			"description": "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "The relative path of a file in the working directory.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

func (rf ReadFile) Call(ctx context.Context, arguments map[string]string) (client.D, error) {
	var readFileInput ReadFileInput
	err := json.Unmarshal([]byte(arguments["path"]), &readFileInput)
	if err != nil {
		return client.D{}, fmt.Errorf("unmarshal: %w", err)
	}

	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return client.D{}, fmt.Errorf("read file: %w", err)
	}

	return client.D{
		"role":    "tool",
		"name":    rf.Name(),
		"content": string(content),
	}, nil
}
