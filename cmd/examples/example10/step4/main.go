// https://ampcode.com/how-to-build-an-agent
//
// This example shows you a final example of the coding agent with support
// to read, list, and edit files.
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
	"errors"
	"fmt"
	"go/format"
	"go/parser"
	"go/token"
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
	Call(ctx context.Context, arguments map[string]any) client.D
}

// =============================================================================

type Agent struct {
	client         *client.SSEClient[client.Chat]
	getUserMessage func() (string, bool)
	tools          map[string]Tool
	toolDocuments  []client.D
}

func NewAgent(sseClient *client.SSEClient[client.Chat], getUserMessage func() (string, bool)) *Agent {
	rf := NewReadFile()
	lf := NewListFiles()
	cf := NewCreateFile()
	gce := NewGoCodeEditor()

	tools := map[string]Tool{
		rf.Name():  rf,
		lf.Name():  lf,
		cf.Name():  cf,
		gce.Name(): gce,
	}

	toolDocs := make([]client.D, 0, len(tools))
	for _, tool := range tools {
		toolDocs = append(toolDocs, tool.ToolDocument())
	}

	return &Agent{
		client:         sseClient,
		getUserMessage: getUserMessage,
		tools:          tools,
		toolDocuments:  toolDocs,
	}
}

var systemPrompt = `You are a helpful coding assistant that has tools to assist
you in coding. 

After you request a tool call, you will receive a JSON document with two fields,
"status" and "data". Always check the "status" field to know if the call "SUCCEED"
or "FAILED". The information you need to respond will be provided under the "data"
field. If the called "FAILED", just inform the user and don't try using the tool
again for the current response.

When reading Go source code always start counting lines of code from the top of
the source code file.`

func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D
	var inToolCall bool

	conversation = append(conversation, client.D{
		"role":    "system",
		"content": systemPrompt,
	})

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
		}

		inToolCall = false

		d := client.D{
			"model":          "qwen3:8b",
			"messages":       conversation,
			"max_tokens":     32768,
			"temperature":    0.1,
			"top_p":          0.1,
			"top_k":          50,
			"stream":         true,
			"tools":          a.toolDocuments,
			"tool_selection": "auto",
			"options":        client.D{"num_ctx": 32768},
		}

		fmt.Print("\u001b[93m\nqwen3\u001b[0m: ")

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
			case "</think>":
				thinking = false
			}

			switch {
			case len(resp.Message.ToolCalls) > 0:
				results := a.callTools(ctx, resp.Message.ToolCalls)
				if len(results) > 0 {
					conversation = append(conversation, results...)
					inToolCall = true
				}

			case resp.Message.Content != "":
				fmt.Print(resp.Message.Content)
				if !thinking && resp.Message.Content != "</think>" {
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

func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) []client.D {
	var resps []client.D

	for _, toolCall := range toolCalls {
		tool, exists := a.tools[toolCall.Function.Name]
		if !exists {
			continue
		}

		fmt.Printf("\u001b[92mtool\u001b[0m: %s(%v)\n", tool.Name(), toolCall.Function.Arguments)

		resp := tool.Call(ctx, toolCall.Function.Arguments)
		resps = append(resps, resp)

		fmt.Printf("%#v\n", resps)
	}

	return resps
}

// =============================================================================

func toolSuccessResponse(toolName string, values ...any) client.D {
	data := make(map[string]any)
	for i := 0; i < len(values); i = i + 2 {
		data[values[i].(string)] = values[i+1]
	}

	info := struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}{
		Status: "SUCCESS",
		Data:   data,
	}

	json, err := json.Marshal(info)
	if err != nil {
		return client.D{
			"role":    "tool",
			"name":    "error",
			"content": `{"status": "FAILED", "data": "error marshaling tool response"}`,
		}
	}

	return client.D{
		"role":    "tool",
		"name":    toolName,
		"content": string(json),
	}
}

func toolErrorResponse(toolName string, err error) client.D {
	data := map[string]any{"error": err.Error()}

	info := struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}{
		Status: "FAILED",
		Data:   data,
	}

	json, err := json.Marshal(info)
	if err != nil {
		return client.D{
			"role":    "tool",
			"name":    "error",
			"content": `{"status": "FAILED", "data": "error marshaling tool response"}`,
		}
	}

	content := string(json)

	fmt.Printf("\n\u001b[92m\ntool\u001b[0m: %s\n", content)

	return client.D{
		"role":    "tool",
		"name":    toolName,
		"content": content,
	}
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
			"name":        rf.name,
			"description": "Read the contents of a given file path or search for files containing a pattern. When searching file contents, returns line numbers where the pattern is found.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "The relative path of a file in the working directory. If pattern is provided, this can be a directory path to search in.",
					},
					"pattern": client.D{
						"type":        "string",
						"description": " The pattern to search for in file contents. If not provided keep it empty. If provided, will search for files containing this pattern instead of reading a specific file.",
					},
				},
				"required": []string{"path", "pattern"},
			},
		},
	}
}

func (rf ReadFile) Call(ctx context.Context, arguments map[string]any) client.D {
	dir := "."
	if arguments["path"] != "" {
		dir = arguments["path"].(string)
	}

	pattern := ""
	if arguments["pattern"] != "" {
		pattern = arguments["pattern"].(string)
	}

	switch pattern {
	case "":
		content, err := os.ReadFile(dir)
		if err != nil {
			return toolErrorResponse(rf.name, err)
		}
		return toolSuccessResponse(rf.name, "file_contents", string(content))

	default:
		var files []any
		err := filepath.Walk(dir, func(filePath string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			relPath, err := filepath.Rel(dir, filePath)
			if err != nil {
				return err
			}

			if strings.Contains(relPath, "zarf") ||
				strings.Contains(relPath, "vendor") ||
				strings.Contains(relPath, ".venv") ||
				strings.Contains(relPath, ".git") {
				return nil
			}

			// Search file contents for pattern
			content, err := os.ReadFile(filePath)
			if err != nil {
				return nil
			}

			lines := strings.Split(string(content), "\n")
			var matchingLines []int

			for i, line := range lines {
				if strings.Contains(strings.ToLower(line), strings.ToLower(pattern)) {
					matchingLines = append(matchingLines, i+1) // Line numbers are 1-indexed
				}
			}

			if len(matchingLines) > 0 {
				fileInfo := map[string]any{
					"file":         relPath,
					"line_numbers": matchingLines,
				}
				files = append(files, fileInfo)
			}

			return nil
		})

		if err != nil {
			return toolErrorResponse(rf.name, err)
		}

		return toolSuccessResponse(rf.name, "files", files)
	}
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
			"name":        lf.name,
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

func (lf ListFiles) Call(ctx context.Context, arguments map[string]any) client.D {
	dir := "."
	if arguments["path"] != "" {
		dir = arguments["path"].(string)
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
		return toolErrorResponse(lf.name, err)
	}

	return toolSuccessResponse(lf.name, "files", files)
}

// =============================================================================

type CreateFile struct {
	name string
}

func NewCreateFile() CreateFile {
	return CreateFile{
		name: "create_file",
	}
}

func (cf CreateFile) Name() string {
	return cf.name
}

func (cf CreateFile) ToolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        cf.name,
			"description": "Create a new file",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "The path to the file",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

func (cf CreateFile) Call(ctx context.Context, arguments map[string]any) client.D {
	filePath := arguments["path"].(string)

	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		return toolErrorResponse(cf.name, errors.New("file already exists"))
	}

	dir := path.Dir(filePath)
	if dir != "." {
		os.MkdirAll(dir, 0755)
	}

	f, err := os.Create(filePath)
	if err != nil {
		return toolErrorResponse(cf.name, err)
	}
	f.Close()

	return toolSuccessResponse(cf.name, "message", "File created successfully")
}

// =============================================================================

type GoCodeEditor struct {
	name string
}

func NewGoCodeEditor() GoCodeEditor {
	return GoCodeEditor{
		name: "golang_code_editor",
	}
}

func (gce GoCodeEditor) Name() string {
	return gce.name
}

func (gce GoCodeEditor) ToolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        gce.name,
			"description": "Edit Golang source code files including adding, replacing, and deleting lines.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "The path to the Golang source code file",
					},
					"line_number": client.D{
						"type":        "integer",
						"description": "The line number for the code change",
					},
					"type_change": client.D{
						"type":        "string",
						"description": "The type of change to make: add, replace, delete",
					},
					"line_change": client.D{
						"type":        "string",
						"description": "The text to add, replace, delete",
					},
				},
				"required": []string{"path", "line_number", "type_change", "line_change"},
			},
		},
	}
}

func (gce GoCodeEditor) Call(ctx context.Context, arguments map[string]any) client.D {
	path := arguments["path"].(string)
	lineNumber := int(arguments["line_number"].(float64))
	typeChange := strings.TrimSpace(arguments["type_change"].(string))
	lineChange := strings.TrimSpace(arguments["line_change"].(string))

	content, err := os.ReadFile(path)
	if err != nil {
		return toolErrorResponse(gce.name, err)
	}

	fset := token.NewFileSet()
	lines := strings.Split(string(content), "\n")

	if lineNumber < 1 || lineNumber > len(lines) {
		return toolErrorResponse(gce.name, fmt.Errorf("line number %d is out of range (1-%d)", lineNumber, len(lines)))
	}

	switch typeChange {
	case "add":
		newLines := make([]string, 0, len(lines)+1)
		newLines = append(newLines, lines[:lineNumber-1]...)
		newLines = append(newLines, lineChange)
		newLines = append(newLines, lines[lineNumber-1:]...)
		lines = newLines

	case "replace":
		lines[lineNumber-1] = lineChange

	case "delete":
		if len(lines) == 1 {
			lines = []string{""}
		} else {
			lines = append(lines[:lineNumber-1], lines[lineNumber:]...)
		}

	default:
		return toolErrorResponse(gce.name, fmt.Errorf("unsupported change type: %s, please inform the user", typeChange))
	}

	modifiedContent := strings.Join(lines, "\n")

	_, err = parser.ParseFile(fset, path, modifiedContent, parser.ParseComments)
	if err != nil {
		return toolErrorResponse(gce.name, fmt.Errorf("syntax error after modification: %s, please inform the user", err))
	}

	formattedContent, err := format.Source([]byte(modifiedContent))
	if err != nil {
		formattedContent = []byte(modifiedContent)
	}

	err = os.WriteFile(path, formattedContent, 0644)
	if err != nil {
		return toolErrorResponse(gce.name, fmt.Errorf("write file: %s", err))
	}

	var action string
	switch typeChange {
	case "add":
		action = fmt.Sprintf("Added line at position %d", lineNumber)
	case "replace":
		action = fmt.Sprintf("Replaced line %d", lineNumber)
	case "delete":
		action = fmt.Sprintf("Deleted line %d", lineNumber)
	}

	return toolSuccessResponse(gce.name, "message", action)
}
