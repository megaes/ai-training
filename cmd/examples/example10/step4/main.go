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
	"path"
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
		NewEditFile(),
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
						fmt.Printf("\n\n\u001b[92m\ntool\u001b[0m: %s", err)
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

		if !inToolCall && len(chunks) > 0 {
			fmt.Print("\n")

			content := strings.Join(chunks, " ")
			content = strings.TrimLeft(content, "\n")

			if content != "" {
				conversation = append(conversation, client.D{
					"role":    "assistant",
					"content": content,
				})
			}
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
					return client.D{}, fmt.Errorf("\n\nERROR: %w", err)
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
						"description": "Relative path to list files from. Defaults to current directory if not provided.",
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

// =============================================================================

type EditFile struct {
	name string
}

func NewEditFile() EditFile {
	return EditFile{
		name: "edit_file",
	}
}

func (ef EditFile) Name() string {
	return ef.name
}

func (ef EditFile) ToolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        ef.Name(),
			"description": "Make edits to a text file. Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other. If the file specified with path doesn't exist, it will be created with a sample hello world for that file type.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "The path to the file",
					},
					"old_str": client.D{
						"type":        "string",
						"description": "Text to search for - must match exactly and must only have one match exactlye",
					},
					"new_str": client.D{
						"type":        "string",
						"description": "Text to replace old_str with",
					},
				},
				"required": []string{"path", "old_str", "new_str"},
			},
		},
	}
}

func (ef EditFile) Call(ctx context.Context, arguments map[string]string) (client.D, error) {
	path := arguments["path"]
	oldStr := arguments["old_str"]
	newStr := arguments["new_str"]

	if path == "" || oldStr == newStr {
		return client.D{}, fmt.Errorf("invalid input parameters")
	}

	content, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return ef.createNewFile(path, newStr)
		}
		return client.D{}, fmt.Errorf("reading file: %w", err)
	}

	if oldStr != "" {
		oldContent := string(content)
		newContent := strings.ReplaceAll(oldContent, oldStr, newStr)

		if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
			return client.D{}, fmt.Errorf("writing file: %w", err)
		}
	}

	fmt.Printf("\n\n\u001b[92m\ntool\u001b[0m: File %s edited successfully", path)

	return client.D{}, nil
}

func (ef EditFile) createNewFile(filePath string, content string) (client.D, error) {
	dir := path.Dir(filePath)
	if dir != "." {
		fmt.Println("FP: ", filePath)
		fmt.Println("DIR: ", dir)
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return client.D{}, fmt.Errorf("creating directory: %w", err)
		}
	}

	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return client.D{}, fmt.Errorf("writing file: %w", err)
	}

	fmt.Printf("\n\n\u001b[92m\ntool\u001b[0m: File %s created successfully", filePath)

	return client.D{}, nil
}
