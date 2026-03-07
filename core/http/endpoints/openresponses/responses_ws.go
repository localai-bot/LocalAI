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
	"github.com/mudler/LocalAI/core/schema"
	"github.com/mudler/LocalAI/pkg/model"
	"github.com/mudler/xlog"
)

const (
	writeWait     = 10 * time.Second
	pongWait      = 60 * time.Second
	pingPeriod    = (pongWait * 9) / 10
	maxMessageSize = 10 * 1024 * 1024
)

type Connection struct {
	conn       *websocket.Conn
	sessionID  string
	responseID string
	previousID string
	createdAt  time.Time
	lastActive time.Time
	mu         sync.Mutex
	hub        *ConnectionHub
	done       chan struct{}
}

type ConnectionHub struct {
	connections map[string]*Connection
	mu          sync.RWMutex
}

var hub = &ConnectionHub{
	connections: make(map[string]*Connection),
}

type LockedWebsocket struct {
	*websocket.Conn
	sync.Mutex
}

func (l *LockedWebsocket) WriteMessage(messageType int, data []byte) error {
	l.Lock()
	defer l.Unlock()
	return l.Conn.WriteMessage(messageType, data)
}

const (
	ServerEventResponseCreated   = "response.created"
	ServerEventResponseProgress  = "response.progress"
	ServerEventResponseDone      = "response.done"
	ServerEventResponseFailed    = "response.failed"
	ServerEventError             = "error"
	ServerEventOutputTextDelta   = "output.text.delta"
	ServerEventOutputTextDone    = "output.text.done"
	ServerEventOutputFunctionCallArgumentsDelta = "output.function_call_arguments.delta"
	ServerEventOutputFunctionCallArgumentsDone    = "output.function_call_arguments.done"
)

type ResponseStatus string

const (
	ResponseStatusInProgress ResponseStatus = "in_progress"
	ResponseStatusCompleted  ResponseStatus = "completed"
	ResponseStatusFailed     ResponseStatus = "failed"
)

type ServerEvent struct {
	Type string `json:"type"`
}

type ResponseCreatedEvent struct {
	ServerEvent
	Response *schema.ORResponseResource `json:"response"`
}

type ResponseProgressEvent struct {
	ServerEvent
	Response *schema.ORResponseResource `json:"response"`
}

type ResponseDoneEvent struct {
	ServerEvent
	Response *schema.ORResponseResource `json:"response"`
}

type ErrorResponse struct {
	ServerEvent
	Error schema.Error `json:"error"`
}

const (
	MessageTypeCreate      = "response.create"
	MessageTypeCancel      = "response.cancel"
	MessageTypeCompact     = "response.compact"
)

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
	Warmup               *bool                                      `json:"generate,omitempty"`
	ReasoningEffort      string                                     `json:"reasoning_effort,omitempty"`
}

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

func ResponsesWebSocket(application *application.Application) echo.HandlerFunc {
	return func(c echo.Context) error {
		conn, err := websocket.Upgrade(c.Response(), c.Request(), nil, 0, 0)
		if err != nil {
			return err
		}
		conn.SetReadLimit(maxMessageSize)

		sessionID := uuid.New().String()
		wsConn := &LockedWebsocket{Conn: conn}

		connection := &Connection{
			conn:        wsConn,
			sessionID:   sessionID,
			createdAt:   time.Now(),
			lastActive:  time.Now(),
			hub:         hub,
			done:        make(chan struct{}),
		}

		hub.Register(connection)
		defer hub.Unregister(sessionID)

		xlog.Info("WebSocket connection established", "sessionID", sessionID)

		go connection.timeoutHandler(60)
		go connection.pingHandler()
		return connection.handleMessages(application)
	}
}

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
			c.conn.Close()
			return err
		}
		c.lastActive = time.Now()

		if messageType == websocket.TextMessage {
			var req ClientRequest
			if err := json.Unmarshal(message, &req); err != nil {
				c.sendError(ServerEventError, "invalid_request", fmt.Sprintf("Failed to parse message: %v", err), "")
				continue
			}
			c.handleClientRequest(application, req)
		}
	}
}

func (c *Connection) handleClientRequest(application *application.Application, req ClientRequest) {
	switch req.Type {
	case MessageTypeCreate:
		c.handleCreate(application, req)
	case MessageTypeCancel:
		c.sendError(ServerEventResponseFailed, "not_implemented", "Cancellation not yet implemented", "")
	case MessageTypeCompact:
		c.sendError(ServerEventResponseFailed, "not_implemented", "Compaction not yet implemented", "")
	default:
		c.sendError(ServerEventError, "invalid_request", fmt.Sprintf("Unknown message type: %s", req.Type), "")
	}
}

func (c *Connection) handleCreate(application *application.Application, req ClientRequest) {
	if req.Warmup != nil && *req.Warmup {
		xlog.Debug("Warmup request received", "sessionID", c.sessionID)
		return
	}

	if req.PreviousResponseID != "" {
		c.previousID = req.PreviousResponseID
		xlog.Debug("Continuation request with previous_response_id", "sessionID", c.sessionID, "previousID", req.PreviousResponseID)
	}

	c.responseID = fmt.Sprintf("resp_%s", uuid.New().String())

	response := &schema.ORResponseResource{
		ID:         c.responseID,
		Object:     "response",
		Status:     ResponseStatusInProgress,
		Model:      req.Model,
		Input:      req.Input,
		Instructions: req.Instructions,
		Metadata:   req.Metadata,
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

	c.sendEvent(ResponseCreatedEvent{
		ServerEvent: ServerEvent{Type: ServerEventResponseCreated},
		Response:    response,
	})

	c.processResponse(application, req, response)
}

func (c *Connection) processResponse(application *application.Application, req ClientRequest, response *schema.ORResponseResource) {
	defer func() {
		if r := recover(); r != nil {
			xlog.Error("Panic in processResponse", "sessionID", c.sessionID, "error", r)
			c.sendError(ServerEventResponseFailed, "internal_error", fmt.Sprintf("Internal error: %v", r), "")
		}
	}()

	modelLoader := application.ModelLoader()
	configLoader := application.ModelConfigLoader()

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

	messages := make([]schema.Message, 0)
	if req.Instructions != "" {
		messages = append(messages, schema.Message{
			Role:          "system",
			StringContent: req.Instructions,
		})
	}
	messages = append(messages, req.Input...)

	if c.previousID != "" {
		xlog.Debug("Previous response ID noted for continuation", "sessionID", c.sessionID, "id", c.previousID)
	}

	predictor, err := model.NewModel(application.Context(), cfg.Path, cfg, modelLoader)
	if err != nil {
		xlog.Error("Failed to create predictor", "sessionID", c.sessionID, "error", err)
		c.sendError(ServerEventError, "model_init_error", "Failed to initialize model", "")
		return
	}
	defer predictor.Delete()

	predictReq := schema.PredictRequest{
		Messages:          messages,
		Stream:            false,
		Temperature:       req.Temperature,
		TopP:              req.TopP,
		MaxTokens:         req.MaxOutputTokens,
		Stop:              req.Stop,
		Tools:             req.Tools,
		ToolChoice:        req.ToolChoice,
		ParallelToolCalls: req.ParallelToolCalls,
	}

	result, err := predictor.Predict(application.Context(), predictReq)
	if err != nil {
		xlog.Error("Prediction failed", "sessionID", c.sessionID, "error", err)
		c.sendError(ServerEventResponseFailed, "prediction_error", fmt.Sprintf("Prediction failed: %v", err), "")
		return
	}

	response.Status = ResponseStatusCompleted
	response.Output = []schema.OutputItem{}

	outputItem := schema.OutputItem{
		Type:   "message",
		ID:     fmt.Sprintf("item_%s", uuid.New().String()),
		Status: "completed",
		Content: []schema.ContentPart{
			{
				Type: "output_text",
				Text: result.Response,
			},
		},
	}
	response.Output = append(response.Output, outputItem)

	c.sendEvent(ResponseProgressEvent{
		ServerEvent: ServerEvent{Type: ServerEventResponseProgress},
		Response:    response,
	})

	c.sendEvent(ResponseDoneEvent{
		ServerEvent: ServerEvent{Type: ServerEventResponseDone},
		Response:    response,
	})

	xlog.Info("Response completed", "sessionID", c.sessionID, "responseID", c.responseID)
}

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
