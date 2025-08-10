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
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/tiktoken"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	url             = "http://localhost:11434/v1/chat/completions"
	model           = "gpt-oss:latest"
	mcpHost         = "localhost:8080"
	maxInputTokens  = 8192
	maxOutputTokens = 8192
	contextWindow   = maxInputTokens + maxOutputTokens
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {

	// -------------------------------------------------------------------------
	// Runs the MCP server locally for our example purposes. This could be
	// replaced with a MCP server that is running in a different process.

	go func() {
		mcpListenAndServe(mcpHost)
	}()

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

	agent, err := NewAgent(getUserMessage)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	return agent.Run(context.TODO())
}

// =============================================================================

// Tool defines the interface that all tools must implement.
type Tool interface {
	Name() string
	Call(ctx context.Context, arguments map[string]any) client.D
}

// =============================================================================

// Agent represents the chat agent that can use tools to perform tasks.
type Agent struct {
	sseClient      *client.SSEClient[client.ChatSSE]
	mcpClient      *mcpClient
	getUserMessage func() (string, bool)
	tke            *tiktoken.Tiktoken
	tools          map[string]Tool
	toolDocuments  []client.D
}

// NewAgent creates a new instance of Agent.
func NewAgent(getUserMessage func() (string, bool)) (*Agent, error) {

	// -------------------------------------------------------------------------
	// Construct the SSE client to make model calls.

	logger := func(ctx context.Context, msg string, v ...any) {
		s := fmt.Sprintf("msg: %s", msg)
		for i := 0; i < len(v); i = i + 2 {
			s = s + fmt.Sprintf(", %s: %v", v[i], v[i+1])
		}
		log.Println(s)
	}

	sseClient := client.NewSSE[client.ChatSSE](logger)

	// -------------------------------------------------------------------------
	// Construct the mcp client.

	mcpClient := newMCPClient()

	// -------------------------------------------------------------------------
	// Construct the tokenizer.

	tke, err := tiktoken.NewTiktoken()
	if err != nil {
		return nil, fmt.Errorf("failed to create tiktoken: %w", err)
	}

	// -------------------------------------------------------------------------
	// Construct the agent.

	tools := map[string]Tool{}

	agent := Agent{
		sseClient:      sseClient,
		mcpClient:      mcpClient,
		getUserMessage: getUserMessage,
		tke:            tke,
		tools:          tools,
		toolDocuments: []client.D{
			NewReadFile(mcpClient, tools),
			NewSearchFiles(tools),
			NewCreateFile(tools),
			NewGoCodeEditor(tools),
		},
	}

	return &agent, nil
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

If you get back results from a tool call, do not verify the results.

Reasoning: high
`

// Run starts the agent and runs the chat loop.
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
			"max_tokens":     maxOutputTokens,
			"temperature":    0.0,
			"top_p":          0.1,
			"top_k":          1,
			"stream":         true,
			"tools":          a.toolDocuments,
			"tool_selection": "auto",
			"options":        client.D{"num_ctx": contextWindow},
		}

		fmt.Printf("\u001b[93m\n%s\u001b[0m: ", model)

		ch := make(chan client.ChatSSE, 100)
		ctx, cancelContext := context.WithTimeout(ctx, time.Minute*5)

		if err := a.sseClient.Do(ctx, http.MethodPost, url, d, ch); err != nil {
			cancelContext()
			fmt.Printf("\n\n\u001b[91mERROR:%s\u001b[0m\n\n", err)
			inToolCall = false
			lastToolCall = nil
			continue
		}

		// ---------------------------------------------------------------------
		// Now we will make a call to the model.

		var chunks []string      // Store the response chunks since we are streaming.
		reasonThinking := false  // GPT models provide a Reasoning field.
		contentThinking := false // Other reasoning models use <think> tags.
		reasonContent = nil      // Reset the reasoning content for this next call.

		fmt.Print("\n")

		// ---------------------------------------------------------------------
		// Process the response which comes in as chunks. So we need to process
		// and save each chunk.

		for resp := range ch {
			switch {

			// Did the model ask us to execute a tool call?
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

		cancelContext()

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

// callTools will lookup a requested tool by name and call it.
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

// addToConversation will add new messages to the conversation history and
// calculate the different tokens used in the conversation and display it to the
// user. It will also check the amount of input tokens currently in history
// and remove the oldest messages if we are over.
func (a *Agent) addToConversation(reasoning []string, conversation []client.D, newMessages ...client.D) []client.D {
	conversation = append(conversation, newMessages...)

	fmt.Print("\n")

	for {
		var totalInput int
		for _, c := range conversation {
			totalInput += a.tke.TokenCount(c["content"].(string))
		}

		r := strings.Join(reasoning, "")
		reasonTokens := a.tke.TokenCount(r)

		totalTokens := totalInput + reasonTokens
		percentage := (float64(totalInput) / float64(maxInputTokens)) * 100
		of := float32(maxInputTokens) / float32(1024)

		fmt.Printf("\u001b[90mTokens Total[%d] Rea[%d] Inp[%d] (%.0f%% of %.0fK Inp tokens)\u001b[0m\n", totalTokens, reasonTokens, totalInput, percentage, of)

		// ---------------------------------------------------------------------
		// Check if we have too many input tokens and start removing messages.

		if totalInput > maxInputTokens {
			fmt.Print("\u001b[90mRemoving conversation history\u001b[0m\n")
			conversation = slices.Delete(conversation, 1, 2)
			continue
		}

		break
	}

	return conversation
}

// =============================================================================

// compareToolCalls will try and detect if the model is asking us to call the
// same tool twice. This function is not accurate because the arguments are in a
// map. We need to fix that.
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

// ReadFile represents a tool that can be used to read the contents of a file.
type ReadFile struct {
	name      string
	mcpClient *mcpClient
	transport *mcp.SSEClientTransport
}

// NewReadFile creates a new instance of the ReadFile tool and loads it
// into the provided tools map.
func NewReadFile(mcpClient *mcpClient, tools map[string]Tool) client.D {
	toolName := "read_file"

	addr := fmt.Sprintf("http://%s/%s", mcpHost, toolName)
	transport := mcp.NewSSEClientTransport(addr, nil)

	rf := ReadFile{
		name:      toolName,
		mcpClient: mcpClient,
		transport: transport,
	}
	tools[rf.name] = &rf

	return rf.toolDocument()
}

// Name returns the name of the tool.
func (rf *ReadFile) Name() string {
	return rf.name
}

// ToolDocument defines the metadata for the tool that is provied to the model.
func (rf *ReadFile) toolDocument() client.D {
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

// Call is the function that is called by the agent to read the contents of a
// file when the model requests the tool with the specified parameters.
func (rf *ReadFile) Call(ctx context.Context, arguments map[string]any) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(rf.name, fmt.Errorf("%s", r))
		}
	}()

	params := &mcp.CallToolParams{
		Name:      rf.name,
		Arguments: arguments,
	}

	results, err := rf.mcpClient.Call(ctx, rf.transport, params)
	if err != nil {
		return toolErrorResponse(rf.name, fmt.Errorf("failed to call tool: %w", err))
	}

	data := results[0].(*mcp.TextContent).Text

	var info struct {
		Data map[string]string `json:"data"`
	}

	if err := json.Unmarshal([]byte(data), &info); err != nil {
		return toolErrorResponse(rf.name, err)
	}

	return toolSuccessResponse(rf.name, "file_contents", info.Data["file_contents"])
}

// =============================================================================
// SearchFiles Tool

// SearchFiles represents a tool that can be used to list files.
type SearchFiles struct {
	name string
}

// NewSearchFiles creates a new instance of the SearchFiles tool and loads it
// into the provided tools map.
func NewSearchFiles(tools map[string]Tool) client.D {
	sf := SearchFiles{
		name: "search_files",
	}
	tools[sf.name] = &sf

	return sf.toolDocument()
}

// Name returns the name of the tool.
func (sf *SearchFiles) Name() string {
	return sf.name
}

// toolDocument defines the metadata for the tool that is provied to the model.
func (sf *SearchFiles) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        sf.name,
			"description": "Search a directory at a given path for files that match a given file name or contain a given string. If no path is provided, search files will look in the current directory.",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"path": client.D{
						"type":        "string",
						"description": "Relative path to search files from. Defaults to current directory if not provided.",
					},
					"filter": client.D{
						"type":        "string",
						"description": "The filter to apply to the file names. It supports golang regex syntax. If not provided, will filtering with take place. If provided, only return files that match the filter.",
					},
					"contains": client.D{
						"type":        "string",
						"description": "A string to search for inside files. It supports golang regex syntax. If not provided, no search will be performed. If provided, only return files that contain the string.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

// Call is the function that is called by the agent to list files when the model
// requests the tool with the specified parameters.
func (sf *SearchFiles) Call(ctx context.Context, arguments map[string]any) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(sf.name, fmt.Errorf("%s", r))
		}
	}()

	dir := "."
	if v, exists := arguments["path"]; exists && v != "" {
		dir = v.(string)
	}

	filter := ""
	if v, exists := arguments["filter"]; exists {
		filter = v.(string)
	}

	contains := ""
	if v, exists := arguments["contains"]; exists {
		contains = v.(string)
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
			strings.Contains(relPath, "libw2v") ||
			strings.Contains(relPath, ".git") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if relPath == "." {
			return nil
		}

		if filter != "" {
			if matched, _ := regexp.MatchString(filter, relPath); !matched {
				return nil
			}
		}

		if contains != "" {
			content, err := os.ReadFile(relPath)
			if err != nil {
				return nil
			}

			if matched, _ := regexp.MatchString(contains, string(content)); !matched {
				return nil
			}
		}

		switch {
		case info.IsDir():
			files = append(files, relPath+"/")

		default:
			files = append(files, relPath)
		}

		return nil
	})

	if err != nil {
		return toolErrorResponse(sf.name, err)
	}

	return toolSuccessResponse(sf.name, "files", files)
}

// =============================================================================
// CreateFile Tool

// CreateFile represents a tool that can be used to create files.
type CreateFile struct {
	name string
}

// NewCreateFile creates a new instance of the CreateFile tool and loads it
// into the provided tools map.
func NewCreateFile(tools map[string]Tool) client.D {
	cf := CreateFile{
		name: "create_file",
	}
	tools[cf.name] = &cf

	return cf.toolDocument()
}

// Name returns the name of the tool.
func (cf *CreateFile) Name() string {
	return cf.name
}

// toolDocument defines the metadata for the tool that is provied to the model.
func (cf *CreateFile) toolDocument() client.D {
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

// Call is the function that is called by the agent to create a file when the model
// requests the tool with the specified parameters.
func (cf *CreateFile) Call(ctx context.Context, arguments map[string]any) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(cf.name, fmt.Errorf("%s", r))
		}
	}()

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

// GoCodeEditor represents a tool that can be used to edit Go source code files.
type GoCodeEditor struct {
	name string
}

// NewGoCodeEditor creates a new instance of the GoCodeEditor tool and loads it
// into the provided tools map.
func NewGoCodeEditor(tools map[string]Tool) client.D {
	gce := GoCodeEditor{
		name: "go_code_editor",
	}
	tools[gce.name] = &gce

	return gce.toolDocument()
}

// Name returns the name of the tool.
func (gce *GoCodeEditor) Name() string {
	return gce.name
}

// toolDocument defines the metadata for the tool that is provied to the model.
func (gce *GoCodeEditor) toolDocument() client.D {
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

// Call is the function that is called by the agent to edit a file when the model
// requests the tool with the specified parameters.
func (gce *GoCodeEditor) Call(ctx context.Context, arguments map[string]any) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(gce.name, fmt.Errorf("%s", r))
		}
	}()

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
