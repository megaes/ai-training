package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/format"
	"go/parser"
	"go/token"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// mcpListenAndServe starts the MCP server for all the tooling we support.
func mcpListenAndServe(host string) {
	fmt.Printf("\nServer: MCP servers serving at %s\n", host)

	fileOperations := mcp.NewServer(&mcp.Implementation{Name: "file_operations", Version: "v1.0.0"}, nil)

	f := func(request *http.Request) *mcp.Server {
		url := request.URL.Path

		switch url {
		case RegisterReadFileTool(fileOperations),
			RegisterSearchFilesTool(fileOperations),
			RegisterCreateFileTool(fileOperations),
			RegisterGoCodeEditorTool(fileOperations):
			return fileOperations

		default:
			return mcp.NewServer(&mcp.Implementation{Name: "unknown_tool", Version: "v1.0.0"}, nil)
		}
	}

	handler := mcp.NewSSEHandler(f)
	fmt.Println(http.ListenAndServe(host, handler))
}

// =============================================================================

// RegisterReadFileTool registers the read_file tool with the given MCP server.
func RegisterReadFileTool(mcpServer *mcp.Server) string {
	const toolName = "tool_read_file"
	const tooDescription = "Read the contents of a given file path or search for files containing a pattern. When searching file contents, returns line numbers where the pattern is found."

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, ReadFileTool)

	return "/" + toolName
}

// ReadFileToolParams represents the parameters for this tool call.
type ReadFileToolParams struct {
	Path string `json:"path" jsonschema:"a possible filter to use"`
}

// ReadFileTool reads the contents of a given file path.
func ReadFileTool(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[ReadFileToolParams]) (*mcp.CallToolResultFor[any], error) {
	dir := "."
	if params.Arguments.Path != "" {
		dir = params.Arguments.Path
	}

	content, err := os.ReadFile(dir)
	if err != nil {
		return nil, err
	}

	info := struct {
		Contents string `json:"contents"`
	}{
		Contents: string(content),
	}

	data, err := json.Marshal(info)
	if err != nil {
		return nil, err
	}

	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(data),
		}},
	}, nil
}

// =============================================================================

// RegisterSearchFilesTool registers the search_files tool with the given MCP server.
func RegisterSearchFilesTool(mcpServer *mcp.Server) string {
	const toolName = "tool_search_files"
	const tooDescription = "Read the contents of a given file path or search for files containing a pattern. When searching file contents, returns line numbers where the pattern is found."

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, SearchFilesTool)

	return "/" + toolName
}

// SearchFilesToolParams represents the parameters for this tool call.
type SearchFilesToolParams struct {
	Path     string `json:"path" jsonschema:"Relative path to search files from. Defaults to current directory if not provided."`
	Filter   string `json:"filter" jsonschema:"The filter to apply to the file names. It supports golang regex syntax. If not provided, will filtering with take place. If provided, only return files that match the filter."`
	Contains string `json:"contains" jsonschema:"A string to search for inside files. It supports golang regex syntax. If not provided, no search will be performed. If provided, only return files that contain the string."`
}

// SearchFilesTool searches for files in a given directory that match a
// given filter and contain a given string.
func SearchFilesTool(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[SearchFilesToolParams]) (*mcp.CallToolResultFor[any], error) {
	dir := "."
	if params.Arguments.Path != "" {
		dir = params.Arguments.Path
	}

	filter := params.Arguments.Filter
	contains := params.Arguments.Contains

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
		return nil, err
	}

	info := struct {
		Files []string `json:"files"`
	}{
		Files: files,
	}

	data, err := json.Marshal(info)
	if err != nil {
		return nil, err
	}

	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(data),
		}},
	}, nil
}

// =============================================================================

// RegisterCreateFileTool registers the search_files tool with the given MCP server.
func RegisterCreateFileTool(mcpServer *mcp.Server) string {
	const toolName = "tool_create_file"
	const tooDescription = "Creates a new file"

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, CreateFileTool)

	return "/" + toolName
}

// CreateFileToolParams represents the parameters for this tool call.
type CreateFileToolParams struct {
	Path string `json:"path" jsonschema:"Relative path and name of the file to create."`
}

// CreateFileTool searches for files in a given directory that match a
// given filter and contain a given string.
func CreateFileTool(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[SearchFilesToolParams]) (*mcp.CallToolResultFor[any], error) {
	filePath := "."
	if params.Arguments.Path != "" {
		filePath = params.Arguments.Path
	}

	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		return nil, err
	}

	dir := path.Dir(filePath)
	if dir != "." {
		os.MkdirAll(dir, 0755)
	}

	f, err := os.Create(filePath)
	if err != nil {
		return nil, err
	}
	f.Close()

	info := struct {
		Status string `json:"status"`
	}{
		Status: "SUCCESS",
	}

	data, err := json.Marshal(info)
	if err != nil {
		return nil, err
	}

	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(data),
		}},
	}, nil
}

// =============================================================================

// RegisterGoCodeEditorTool registers the go_code_editor tool with the given MCP server.
func RegisterGoCodeEditorTool(mcpServer *mcp.Server) string {
	const toolName = "tool_go_code_editor"
	const tooDescription = "Edit Golang source code files including adding, replacing, and deleting lines."

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, GoCodeEditorTool)

	return "/" + toolName
}

// GoCodeEditorToolParams represents the parameters for this tool call.
type GoCodeEditorToolParams struct {
	Path       string `json:"path" jsonschema:"Relative path and name of the file to create."`
	LineNumber int    `json:"line_number" jsonschema:"Relative path and name of the Golang file"`
	TypeChange string `json:"type_change" jsonschema:"Type of change to make to the file."`
	LineChange string `json:"line_change" jsonschema:"Line of code to add, replace, or delete."`
}

// GoCodeEditorTool can make add, updates, and deletes to go code.
func GoCodeEditorTool(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[GoCodeEditorToolParams]) (*mcp.CallToolResultFor[any], error) {
	path := "."
	if params.Arguments.Path != "" {
		path = params.Arguments.Path
	}

	lineNumber := params.Arguments.LineNumber
	typeChange := strings.TrimSpace(params.Arguments.TypeChange)
	lineChange := strings.TrimSpace(params.Arguments.LineChange)

	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	fset := token.NewFileSet()
	lines := strings.Split(string(content), "\n")

	if lineNumber < 1 || lineNumber > len(lines) {
		return nil, fmt.Errorf("line number %d is out of range (1-%d)", lineNumber, len(lines))
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
		return nil, fmt.Errorf("unsupported change type: %s, please inform the user", typeChange)
	}

	modifiedContent := strings.Join(lines, "\n")

	_, err = parser.ParseFile(fset, path, modifiedContent, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("syntax error after modification: %s, please inform the user", err)
	}

	formattedContent, err := format.Source([]byte(modifiedContent))
	if err != nil {
		formattedContent = []byte(modifiedContent)
	}

	err = os.WriteFile(path, formattedContent, 0644)
	if err != nil {
		return nil, fmt.Errorf("write file: %s", err)
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

	info := struct {
		Message string `json:"message"`
	}{
		Message: action,
	}

	data, err := json.Marshal(info)
	if err != nil {
		return nil, err
	}

	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(data),
		}},
	}, nil
}
