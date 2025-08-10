package main

import (
	"context"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// mcpClient is a client for the MCP server.
type mcpClient struct {
	host   string
	client *mcp.Client
}

// newMCPClient constructs a new MCP client.
func newMCPClient() *mcpClient {
	client := mcp.NewClient(&mcp.Implementation{Name: "mcp-client", Version: "v1.0.0"}, nil)

	return &mcpClient{
		client: client,
	}
}

// Call executes an MCP tool call using the provided transport and parameters.
func (cln *mcpClient) Call(ctx context.Context, transport *mcp.SSEClientTransport, params *mcp.CallToolParams) ([]mcp.Content, error) {
	fmt.Print("\u001b[92mtool: connecting to MCP Server\u001b[0m\n")

	session, err := cln.client.Connect(ctx, transport)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to MCP server: %w", err)
	}
	defer session.Close()

	fmt.Printf("\u001b[92mtool: calling tool: %s\u001b[0m\n\n", params.Name)

	res, err := session.CallTool(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to call tool: %w", err)
	}

	if res.IsError {
		return nil, fmt.Errorf("tool call failed: %s", res.Content)
	}

	return res.Content, nil
}
