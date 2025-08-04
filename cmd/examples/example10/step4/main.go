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
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
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
		NewListFiles(),
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
						fmt.Print(err.Error())
						continue
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
				fmt.Printf("\u001b[92m\ntool\u001b[0m: %s(%s)", tool.Name(), toolCall.Function.Arguments)

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
	content, err := os.ReadFile(arguments["path"])
	if err != nil {
		return client.D{}, fmt.Errorf("read file: %w", err)
	}

	return client.D{
		"role":    "tool",
		"name":    rf.Name(),
		"content": string(content),
	}, nil
}

// =============================================================================

type ListFiles struct {
	name string
}

func NewListFiles() ListFiles {
	return ListFiles{
		name: "list_files",
	}
}

func (lf ListFiles) Name() string {
	return lf.name
}

func (lf ListFiles) ToolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        lf.Name(),
			"description": "List files and directories at a given path. If no path is provided, lists files in the current directory.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "The relative path of the directory.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

func (lf ListFiles) Call(ctx context.Context, arguments map[string]string) (client.D, error) {
	dir := "."
	if arguments["path"] != "" {
		dir = arguments["path"]
	}

	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if strings.Contains(relPath, "zarf") ||
			strings.Contains(relPath, "vendor") ||
			strings.Contains(relPath, ".venv") ||
			strings.Contains(relPath, ".git") {
			return nil
		}

		if relPath != "." {
			if info.IsDir() {
				files = append(files, relPath+"/")
			} else {
				files = append(files, relPath)
			}
		}
		return nil
	})

	if err != nil {
		return client.D{}, err
	}

	return client.D{
		"role":    "tool",
		"name":    lf.Name(),
		"content": strings.Join(files, "\n"),
	}, nil
}
