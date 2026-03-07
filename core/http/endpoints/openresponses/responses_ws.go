package openresponses

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/labstack/echo/v4"
	"github.com/mudler/LocalAI/core/application"
	"github.com/mudler/LocalAI/core/config"
	"github.com/mudler/LocalAI/core/http/middleware"
	"github.com/mudler/LocalAI/core/schema"
	"github.com/mudler/LocalAI/pkg/model"
	"github.com/mudler/LocalAI/pkg/reasoning"
	"github.com/mudler/xlog"
)

const (
	// Time allowed to write a message to the peer
	writeWait = 10 * time.Second

	// Time allowed to read the next pong message from the peer
	pongWait = 60 * time.Second

	// Send pings to peer with this period. Must be less than pongWait
	pingPeriod = (pongWait * 9) / 10

	// Maximum message size allowed from peer
	maxMessageSize = 10 * 1024 * 1024

	// Connection timeout in minutes (60 minutes)
	connectionTimeoutMinutes = 60
)

// Connection represents a WebSocket connection to the Responses API
type Connection struct {
	conn        *websocket.Conn
	sessionID   string
	responseID  string
	previousID  string
	createdAt   time.Time
	lastActive  time.Time
	mu          sync.Mutex
	inFlight    bool
	done        chan struct{}
	hub         *ConnectionHub
}

// ConnectionHub manages all active WebSocket connections
type ConnectionHub struct {
	connections map[string]*Connection
	mu          sync.RWMutex
}

var hub = &ConnectionHub{
	connections: make(map[string]*Connection),
}

// LockedWebsocket wraps a websocket connection with a mutex for safe concurrent writes
type LockedWebsocket struct {
	*websocket.Conn
	sync.Mutex
}

func (l *LockedWebsocket) WriteMessage(messageType int, data []byte) error {
	l.Lock()
	defer l.Unlock()
	return l.Conn.WriteMessage(messageType, data)
}

// Message types for WebSocket protocol
const (
	MessageTypeText = "text"
	MessageTypeJSON = "json"
)

// ServerEvent types
const (
	ServerEventResponseCreated   = "response.created"
	ServerEventResponseProgress  = "response.progress"
	ServerEventResponseDone      = "response.done"
	ServerEventResponseFailed    = "response.failed"
	ServerEventError             = "error"
	ServerEventInputSpeechDone   = "input.speech.done"
	ServerEventInputTextDone     = "input.text.done"
	ServerEventInputContentDone  = "input.content.done"
	ServerEventOutputItemAdded   = "output.item.added"
	ServerEventOutputItemDone    = "output.item.done"
	ServerEventOutputTextDelta   = "output.text.delta"
	ServerEventOutputTextDone    = "output.text.done"
	ServerEventOutputFunctionCallArgumentsDelta = "output.function_call_arguments.delta"
	ServerEventOutputFunctionCallArgumentsDone    = "output.function_call_arguments.done"
	ServerEventOutputAudioTranscriptDelta   = "output.audio_transcript.delta"
	ServerEventOutputAudioTranscriptDone    = "output.audio_transcript.done"
	ServerEventOutputAudioDelta             = "output.audio.delta"
	ServerEventOutputAudioDone              = "output.audio.done"
)

// ResponseStatus represents the status of a response
type ResponseStatus string

const (
	ResponseStatusInProgress ResponseStatus = "in_progress"
	ResponseStatusCompleted  ResponseStatus = "completed"
	ResponseStatusCancelled  ResponseStatus = "cancelled"
	ResponseStatusFailed     ResponseStatus = "failed"
)

// ServerEvent represents a server-sent event
type ServerEvent struct {
	Type string `json:"type"`
}

// ResponseCreatedEvent represents response.created event
type ResponseCreatedEvent struct {
	ServerEvent
	Response *schema.ORResponseResource `json:"response"`
}

// ResponseProgressEvent represents response.progress event
type ResponseProgressEvent struct {
	ServerEvent
	Response *schema.ORResponseResource `json:"response"`
}

// ResponseDoneEvent represents response.done event
type ResponseDoneEvent struct {
	ServerEvent
	Response *schema.ORResponseResource `json:"response"`
}

// ResponseFailedEvent represents response.failed event
type ResponseFailedEvent struct {
	ServerEvent
	Error *schema.Error `json:"error"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	ServerEvent
	Error schema.Error `json:"error"`
}

// ClientMessage types
const (
	MessageTypeCreate      = "response.create"
	MessageTypeCancel      = "response.cancel"
	MessageTypeInputCommit = "input.commit"
	MessageTypeInputDelete = "input.delete"
	MessageTypeInputAdd    = "input.add"
	MessageTypeCompact     = "response.compact"
)

// ClientRequest represents a client request message
type ClientRequest struct {
	Type                 string                                     `json:"type"`
	Model                string                                     `json:"model,omitempty"`
	Input                []schema.Message                           `json:"input,omitempty"`
	PreviousResponseID   string                                     `json:"previous_response_id,omitempty"`
	Instructions         string                                     `json:"instructions,omitempty"`
	Metadata             string                                     `json:"metadata,omitempty"`
	Store                *bool                                      `json:"store,omitempty"`
	Temperature          float64                                    `json:"temperature,omitempty"`
	MaxOutputTokens      *int                                       `json:"max_output_tokens,omitempty"`
	TopP                 float64                                    `json:"top_p,omitempty"`
	Stop                 []string                                   `json:"stop,omitempty"`
	Tools                []schema.Tool                              `json:"tools,omitempty"`
	ToolChoice           interface{}                                `json:"tool_choice,omitempty"`
	ParallelToolCalls    *bool                                      `json:"parallel_tool_calls,omitempty"`
	Background           *bool                                      `json:"background,omitempty"`
	Warmup               *bool                                      `json:"generate,omitempty,omitempty"` // typo in OpenAI spec
	ReasoningEffort      string                                     `json:"reasoning_effort,omitempty"`
	Services             []schema.ServiceConfig                     `json:"services,omitempty"`
	ServicePriorities    []string                                   `json:"service_priorities,omitempty"`
}

// ConnectionHub methods
func (h *ConnectionHub) Register(conn *Connection) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.connections[conn.sessionID] = conn
}

func (h *ConnectionHub) Unregister(sessionID string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if conn, ok := h.connections[sessionID]; ok {
		conn.conn.Close()
		delete(h.connections, sessionID)
	}
}

func (h *ConnectionHub) Get(sessionID string) (*Connection, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	conn, ok := h.connections[sessionID]
	return conn, ok
}

// WebSocket connection handler
func ResponsesWebSocket(application *application.Application) echo.HandlerFunc {
	return func(c echo.Context) error {
		// Upgrade to WebSocket
		conn, err := websocket.Upgrade(c.Response(), c.Request(), nil, 0, 0)
		if err != nil {
			return err
		}

		// Set maximum message size
		conn.SetReadLimit(maxMessageSize)

		// Create session ID
		sessionID := uuid.New().String()

		// Create connection
		wsConn := &LockedWebsocket{Conn: conn}

		connection := &Connection{
			conn:        wsConn,
			sessionID:   sessionID,
			createdAt:   time.Now(),
			lastActive:  time.Now(),
			hub:         hub,
			done:        make(chan struct{}),
		}

		// Register connection
		hub.Register(connection)
		defer hub.Unregister(sessionID)

		xlog.Info("WebSocket connection established", "sessionID", sessionID)

		// Start timeout goroutine
		go connection.timeoutHandler(connectionTimeoutMinutes)

		// Start ping handler
		go connection.pingHandler()

		// Handle messages
		return connection.handleMessages(application)
	}
}

// timeoutHandler enforces connection timeout
func (c *Connection) timeoutHandler(minutes int) {
	timeout := time.Duration(minutes) * time.Minute
	ticker := time.NewTicker(timeout)
	defer ticker.Stop()

	select {
	case <-ticker.C:
		xlog.Info("Connection timeout", "sessionID", c.sessionID)
		c.conn.Close()
		close(c.done)
	case <-c.done:
		return
	}
}

// pingHandler sends periodic pings
func (c *Connection) pingHandler() {
	ticker := time.NewTicker(pingPeriod)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				xlog.Info("WebSocket ping failed", "sessionID", c.sessionID, "error", err)
				c.conn.Close()
				return
			}
		case <-c.done:
			return
		}
	}
}

// handleMessages processes incoming WebSocket messages
func (c *Connection) handleMessages(application *application.Application) error {
	c.conn.SetReadLimit(maxMessageSize)
	c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(pongWait))
		c.lastActive = time.Now()
		return nil
	})

	for {
		messageType, message, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				xlog.Error("WebSocket read error", "sessionID", c.sessionID, "error", err)
			}
			c.conn.Close()
			return err
		}

		// Update last active time
		c.lastActive = time.Now()

		if messageType == websocket.TextMessage {
			// Process the message
			var req ClientRequest
			if err := json.Unmarshal(message, &req); err != nil {
				c.sendError(ServerEventError, "invalid_request", fmt.Sprintf("Failed to parse message: %v", err), "")
				continue
			}

			c.handleClientRequest(application, req)
		}
	}
}

// handleClientRequest processes different message types
func (c *Connection) handleClientRequest(application *application.Application, req ClientRequest) {
	switch req.Type {
	case MessageTypeCreate:
		c.handleCreate(application, req)
	case MessageTypeCancel:
		c.handleCancel(req)
	case MessageTypeCompact:
		c.handleCompact(application, req)
	default:
		c.sendError(ServerEventError, "invalid_request", fmt.Sprintf("Unknown message type: %s", req.Type), "")
	}
}

// handleCreate processes response.create message
func (c *Connection) handleCreate(application *application.Application, req ClientRequest) {
	if req.Warmup != nil && *req.Warmup {
		// Warmup mode - just acknowledge without processing
		xlog.Debug("Warmup request received", "sessionID", c.sessionID)
		return
	}

	// Check for previous_response_id
	if req.PreviousResponseID != "" {
		// Validate previous response exists
		// For WebSocket mode, we may need to store responses in connection-local cache
		// or global store
		xlog.Debug("Continuation request with previous_response_id", "sessionID", c.sessionID, "previousID", req.PreviousResponseID)
		c.previousID = req.PreviousResponseID
	}

	// Create response
	c.responseID = fmt.Sprintf("resp_%s", uuid.New().String())

	// Create response resource
	response := &schema.ORResponseResource{
		ID:       c.responseID,
		Object:   "response",
		Status:   ResponseStatusInProgress,
		Model:    req.Model,
		Input:    req.Input,
		Instructions: req.Instructions,
		Metadata: req.Metadata,
		Store:    false,
	}

	if req.Store != nil {
		response.Store = *req.Store
	}

	if req.MaxOutputTokens != nil {
		response.MaxOutputTokens = *req.MaxOutputTokens
	}

	if req.Temperature > 0 {
		response.Temperature = req.Temperature
	}

	if req.TopP > 0 {
		response.TopP = req.TopP
	}

	if req.ParallelToolCalls != nil {
		response.ParallelToolCalls = *req.ParallelToolCalls
	}

	if len(req.Tools) > 0 {
		response.Tools = req.Tools
	}

	if req.ToolChoice != nil {
		response.ToolChoice = req.ToolChoice
	}

	// Send response.created event
	c.sendEvent(ResponseCreatedEvent{
		ServerEvent: ServerEvent{Type: ServerEventResponseCreated},
		Response:    response,
	})

	// Process the request synchronously (could be made async with background flag)
	c.processResponse(application, req, response)
}

// processResponse executes the model inference
func (c *Connection) processResponse(application *application.Application, req ClientRequest, response *schema.ORResponseResource) {
	defer func() {
		if r := recover(); r != nil {
			xlog.Error("Panic in processResponse", "sessionID", c.sessionID, "error", r)
			c.sendError(ServerEventResponseFailed, "internal_error", fmt.Sprintf("Internal error: %v", r), "")
		}
	}()

	// Get model loader and config
	modelLoader := application.ModelLoader()
	configLoader := application.ModelConfigLoader()
	evaluator := application.TemplatesEvaluator()

	// Load model config
	cfg, err := configLoader.LoadModelConfigFileByNameDefaultOptions(req.Model, application.ApplicationConfig())
	if err != nil {
		xlog.Error("Failed to load model config", "sessionID", c.sessionID, "error", err)
		c.sendError(ServerEventError, "model_load_error", "Failed to load model config", "")
		return
	}

	if cfg == nil {
		xlog.Error("Model config not found", "sessionID", c.sessionID, "model", req.Model)
		c.sendError(ServerEventError, "model_not_found", "Model not found", "")
		return
	}

	// Convert messages to internal format
	messages := make([]schema.Message, 0)

	// Add instructions as system message if provided
	if req.Instructions != "" {
		messages = append(messages, schema.Message{
			Role:          "system",
			StringContent: req.Instructions,
		})
	}

	// Add input messages
	messages = append(messages, req.Input...)

	// Convert previous response to messages if previous_response_id was provided
	if c.previousID != "" {
		// TODO: Retrieve previous response from store and convert to messages
		// For now, we'll just note it for continuation
		xlog.Debug("Previous response ID noted for continuation", "sessionID", c.sessionID, "id", c.previousID)
	}

	// Create predictor
	predictor, err := model.NewModel(application.Context(), cfg.Path, cfg, modelLoader)
	if err != nil {
		xlog.Error("Failed to create predictor", "sessionID", c.sessionID, "error", err)
		c.sendError(ServerEventError, "model_init_error", "Failed to initialize model", "")
		return
	}
	defer predictor.Delete()

	// Build request
	predictReq := schema.PredictRequest{
		Messages:          messages,
		Stream:            false, // WebSocket handles streaming manually
		Temperature:       req.Temperature,
		TopP:              req.TopP,
		MaxTokens:         req.MaxOutputTokens,
		Stop:              req.Stop,
		Tools:             req.Tools,
		ToolChoice:        req.ToolChoice,
		ParallelToolCalls: req.ParallelToolCalls,
	}

	// Execute prediction
	result, err := predictor.Predict(application.Context(), predictReq)
	if err != nil {
		xlog.Error("Prediction failed", "sessionID", c.sessionID, "error", err)
		c.sendError(ServerEventResponseFailed, "prediction_error", fmt.Sprintf("Prediction failed: %v", err), "")
		return
	}

	// Update response status
	response.Status = ResponseStatusCompleted
	response.Output = []schema.OutputItem{}

	// Convert result to output items
	outputItem := schema.OutputItem{
		Type:  "message",
		ID:    fmt.Sprintf("item_%s", uuid.New().String()),
		Status: "completed",
		Content: []schema.ContentPart{
			{
				Type: "output_text",
				Text: result.Response,
			},
		},
	}
	response.Output = append(response.Output, outputItem)

	// Send response.progress event
	c.sendEvent(ResponseProgressEvent{
		ServerEvent: ServerEvent{Type: ServerEventResponseProgress},
		Response:    response,
	})

	// Send response.done event
	c.sendEvent(ResponseDoneEvent{
		ServerEvent: ServerEvent{Type: ServerEventResponseDone},
		Response:    response,
	})

	xlog.Info("Response completed", "sessionID", c.sessionID, "responseID", c.responseID)
}

// handleCancel processes response.cancel message
func (c *Connection) handleCancel(req ClientRequest) {
	xlog.Info("Cancel request received", "sessionID", c.sessionID)
	// TODO: Implement cancellation logic
	c.sendError(ServerEventResponseFailed, "not_implemented", "Cancellation not yet implemented", "")
}

// handleCompact processes response.compact message
func (c *Connection) handleCompact(application *application.Application, req ClientRequest) {
	xlog.Info("Compact request received", "sessionID", c.sessionID)
	// TODO: Implement compaction logic
	c.sendError(ServerEventResponseFailed, "not_implemented", "Compaction not yet implemented", "")
}

// sendEvent sends a server-sent event to the client
func (c *Connection) sendEvent(event interface{}) {
	data, err := json.Marshal(event)
	if err != nil {
		xlog.Error("Failed to marshal event", "error", err)
		return
	}

	c.conn.SetWriteDeadline(time.Now().Add(writeWait))
	if err := c.conn.WriteMessage(websocket.TextMessage, data); err != nil {
		xlog.Error("Failed to write event", "error", err)
		c.conn.Close()
	}
}

// sendError sends an error event to the client
func (c *Connection) sendError(eventType, errorCode, message, param string) {
	err := schema.Error{
		Code:    errorCode,
		Message: message,
		Param:   param,
		Type:    "error",
	}

	event := ErrorResponse{
		ServerEvent: ServerEvent{Type: eventType},
		Error:       err,
	}

	c.sendEvent(event)
}
