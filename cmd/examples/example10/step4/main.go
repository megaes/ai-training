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
	"io/fs"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/tiktoken"
)

const (
	url           = "http://localhost:11434/v1/chat/completions"
	model         = "gpt-oss:latest"
	contextWindow = 168 * 1024 // 168K tokens
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	// -------------------------------------------------------------------------
	// Declare a function that can accept user input which the agent will use
	// when it's the users turn.

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	// -------------------------------------------------------------------------
	// Construct the logger, client to talk to the model, and the agent. Then
	// start the agent.

	logger := func(ctx context.Context, msg string, v ...any) {
		s := fmt.Sprintf("msg: %s", msg)
		for i := 0; i < len(v); i = i + 2 {
			s = s + fmt.Sprintf(", %s: %v", v[i], v[i+1])
		}
		log.Println(s)
	}

	cln := client.NewSSE[client.Chat](logger)

	agent, err := NewAgent(cln, getUserMessage)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

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
	tke            *tiktoken.Tiktoken
}

func NewAgent(sseClient *client.SSEClient[client.Chat], getUserMessage func() (string, bool)) (*Agent, error) {
	// -------------------------------------------------------------------------
	// Construct the tools and initialize all the tool support.

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

	// -------------------------------------------------------------------------
	// Construct the tokenizer.

	tke, err := tiktoken.NewTiktoken()
	if err != nil {
		return nil, fmt.Errorf("failed to create tiktoken: %w", err)
	}

	// -------------------------------------------------------------------------
	// Construct the agent.

	a := Agent{
		client:         sseClient,
		getUserMessage: getUserMessage,
		tools:          tools,
		toolDocuments:  toolDocs,
		tke:            tke,
	}

	return &a, nil
}

// The system prompt for the model so it behaves as expected.
var systemPrompt = `You are a helpful coding assistant that has tools to assist
you in coding.

After you request a tool call, you will receive a JSON document with two fields,
"status" and "data". Always check the "status" field to know if the call "SUCCEED"
or "FAILED". The information you need to respond will be provided under the "data"
field. If the called "FAILED", just inform the user and don't try using the tool
again for the current response.

When reading Go source code always start counting lines of code from the top of
the source code file.

Reasoning: high
`

func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D        // History of the conversation
	var reasonContent []string         // Reasoning content per model call
	var inToolCall bool                // Need to know we are inside a tool call request
	var lastToolCall []client.ToolCall // Last tool call to identify call dups

	conversation = append(conversation, client.D{
		"role":    "system",
		"content": systemPrompt,
	})

	fmt.Printf("Chat with %s (use 'ctrl-c' to quit)\n", model)

	for {
		// ---------------------------------------------------------------------
		// If we are not in a tool call then we can ask the user
		// to provide their next question or request.

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

		// ---------------------------------------------------------------------
		// Now we will make a call to the model, we could be responding to a
		// tool call or providing a user request.

		d := client.D{
			"model":          model,
			"messages":       conversation,
			"max_tokens":     contextWindow,
			"temperature":    0.0,
			"top_p":          0.1,
			"top_k":          1,
			"stream":         true,
			"tools":          a.toolDocuments,
			"tool_selection": "auto",
			"options":        client.D{"num_ctx": contextWindow},
		}

		fmt.Printf("\u001b[93m\n%s\u001b[0m: ", model)

		ch := make(chan client.Chat, 100)
		if err := a.client.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			return fmt.Errorf("do: %w", err)
		}

		// ---------------------------------------------------------------------
		// Now we will make a call to the model

		var chunks []string      // Store the content chunks since we are streaming
		reasonThinking := false  // GPT models provide a Reasoning field
		contentThinking := false // Other reasoning model use <think> tags
		reasonContent = nil      // Reset the reasoning content for this next call

		fmt.Print("\n")

		// ---------------------------------------------------------------------
		// Process the response which comes in as chunks. So we need to process
		// and save each chunk.

		for resp := range ch {
			// -----------------------------------------------------------------
			// Did the model ask us to execute a tool call?
			switch {
			case len(resp.Choices[0].Delta.ToolCalls) > 0:
				fmt.Print("\n\n")

				result := compareToolCalls(lastToolCall, resp.Choices[0].Delta.ToolCalls)
				if len(result) > 0 {
					conversation = a.addToConversation(reasonContent, conversation, result)
					inToolCall = true
					continue
				}

				results := a.callTools(ctx, resp.Choices[0].Delta.ToolCalls)
				if len(results) > 0 {
					conversation = a.addToConversation(reasonContent, conversation, results...)
					inToolCall = true
					lastToolCall = resp.Choices[0].Delta.ToolCalls
				}

			// -----------------------------------------------------------------
			// Did we get content? With some models a <think> tag could exist to
			// indicate reasoning. We need to filter that out and display it as
			// a different color.
			case resp.Choices[0].Delta.Content != "":
				if reasonThinking {
					reasonThinking = false
					fmt.Print("\n\n")
				}

				switch resp.Choices[0].Delta.Content {
				case "<think>":
					contentThinking = true
					continue
				case "</think>":
					contentThinking = false
					continue
				}

				switch {
				case !contentThinking:
					fmt.Print(resp.Choices[0].Delta.Content)
					chunks = append(chunks, resp.Choices[0].Delta.Content)

				case contentThinking:
					reasonContent = append(reasonContent, resp.Choices[0].Delta.Content)
					fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Content)
				}

				lastToolCall = nil

			// -----------------------------------------------------------------
			// Did we get reasoning content? ChatGPT models provide reasoning in
			// the Delta.Reasoning field. Display it as a different color.
			case resp.Choices[0].Delta.Reasoning != "":
				reasonThinking = true

				if len(reasonContent) == 0 {
					fmt.Print("\n")
				}

				reasonContent = append(reasonContent, resp.Choices[0].Delta.Reasoning)
				fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
			}
		}

		// ---------------------------------------------------------------------
		// We processed all the chunks from the response so we need to add
		// this to the conversation history.

		if !inToolCall && len(chunks) > 0 {
			fmt.Print("\n")

			content := strings.Join(chunks, " ")
			content = strings.TrimLeft(content, "\n")

			if content != "" {
				conversation = a.addToConversation(reasonContent, conversation, client.D{
					"role":    "assistant",
					"content": content,
				})
			}
		}
	}

	return nil
}

// Iterate over all the tool call requests and execute the tools. It's been
// my experience we get a single call 100% of the time.
func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) []client.D {
	var resps []client.D

	for _, toolCall := range toolCalls {
		tool, exists := a.tools[toolCall.Function.Name]
		if !exists {
			continue
		}

		fmt.Printf("\u001b[92mtool\u001b[0m: %s(%v)\n", toolCall.Function.Name, toolCall.Function.Arguments)

		resp := tool.Call(ctx, toolCall.Function.Arguments)
		resps = append(resps, resp)

		fmt.Printf("%#v\n", resps)
	}

	return resps
}

// We need to calculate the different tokens used in the conversation and
// display it to the user. We will use this as well to add history to the
// conversation.
func (a *Agent) addToConversation(reasoning []string, conversation []client.D, d ...client.D) []client.D {
	conversation = append(conversation, d...)

	var sysTokens int
	var inputTokens int
	var outputTokens int

	for _, c := range conversation {
		switch c["role"].(string) {
		case "system":
			sysTokens += a.tke.TokenCount(c["content"].(string))

		case "user", "tool":
			inputTokens += a.tke.TokenCount(c["content"].(string))

		case "assistant":
			outputTokens += a.tke.TokenCount(c["content"].(string))
		}
	}

	r := strings.Join(reasoning, "")
	reasonTokens := a.tke.TokenCount(r)

	totalTokens := sysTokens + inputTokens + outputTokens + reasonTokens
	percentage := (float64(totalTokens) / float64(contextWindow)) * 100

	fmt.Printf("\n\u001b[90mTokens Sys[%d] Inp[%d] Out[%d] Rea[%d] Tot[%d] (%.0f%% of 168K)\u001b[0m\n", sysTokens, inputTokens, outputTokens, reasonTokens, totalTokens, percentage)

	return conversation
}

// =============================================================================

// We want to try and detect if the model is asking us to call the same tool
// twice. This function is not accurate because the arguments are in a map. We
// need to fix that.
func compareToolCalls(last []client.ToolCall, current []client.ToolCall) client.D {
	if len(last) != len(current) {
		return client.D{}
	}

	for i := range last {
		if last[i].Function.Name != current[i].Function.Name {
			return client.D{}
		}

		if fmt.Sprintf("%v", last[i].Function.Arguments) != fmt.Sprintf("%v", current[i].Function.Arguments) {
			return client.D{}
		}
	}

	fmt.Printf("\u001b[92mtool\u001b[0m: %s(%v)\n", current[0].Function.Name, current[0].Function.Arguments)
	fmt.Printf("\u001b[92mtool\u001b[0m: %s\n", "Same tool call")

	return toolErrorResponse(current[0].Function.Name, errors.New("data already provided in a previous response, please review the conversation history"))
}

// toolSuccessResponse returns a successful structured tool response.
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

// toolErrorResponse returns a failed structured tool response.
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
// ReadFile Tool

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
				},
				"required": []string{"path"},
			},
		},
	}
}

func (rf ReadFile) Call(ctx context.Context, arguments map[string]any) client.D {
	dir := "."
	if arguments["path"] != "" {
		dir = arguments["path"].(string)
	}

	content, err := os.ReadFile(dir)
	if err != nil {
		return toolErrorResponse(rf.name, err)
	}

	return toolSuccessResponse(rf.name, "file_contents", string(content))
}

// =============================================================================
// ListFiles Tool

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
					"extension": client.D{
						"type":        "string",
						"description": "The file extension to filter by. If not provided, will list all files. If provided, will only list files with the given extension.",
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
	err := filepath.WalkDir(dir, func(path string, info fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, filepath.SkipDir) {
				return nil
			}
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if strings.Contains(relPath, "zarf") ||
			strings.Contains(relPath, "vendor") ||
			strings.Contains(relPath, ".venv") ||
			strings.Contains(relPath, ".idea") ||
			strings.Contains(relPath, ".vscode") ||
			strings.Contains(relPath, ".git") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if relPath == "." {
			return nil
		}

		switch {
		case info.IsDir():
			files = append(files, relPath+"/")

		default:
			if arguments["extension"] != "" {
				if !strings.HasSuffix(relPath, arguments["extension"].(string)) {
					return nil
				}
			}
			files = append(files, relPath)
		}

		return nil
	})

	if err != nil {
		return toolErrorResponse(lf.name, err)
	}

	return toolSuccessResponse(lf.name, "files", files)
}

// =============================================================================
// CreateFile Tool

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
// GoCodeEditor Tool

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
