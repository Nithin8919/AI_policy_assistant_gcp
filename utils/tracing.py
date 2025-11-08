"""
Tracing utility - Distributed tracing for request flow
Tracks request progression through the LangGraph pipeline
"""
from typing import Dict, Any, Optional
from datetime import datetime
import uuid


def create_trace_context(request_id: str) -> Dict[str, Any]:
    """
    Create a tracing context for a request
    
    Args:
        request_id: Unique request identifier
    
    Returns:
        Trace context dictionary
    """
    return {
        "request_id": request_id,
        "trace_id": str(uuid.uuid4()),
        "start_time": datetime.utcnow().isoformat(),
        "spans": []
    }


class Span:
    """Represents a traced operation span"""
    
    def __init__(self, trace_ctx: Dict[str, Any], name: str, metadata: Optional[Dict[str, Any]] = None):
        self.trace_ctx = trace_ctx
        self.span_id = str(uuid.uuid4())
        self.name = name
        self.metadata = metadata or {}
        self.start_time = datetime.utcnow()
        self.end_time = None
        self.duration_ms = None
        self.status = "pending"
        self.error = None
    
    def __enter__(self):
        """Start the span"""
        span_data = {
            "span_id": self.span_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "metadata": self.metadata
        }
        self.trace_ctx["spans"].append(span_data)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the span"""
        self.end_time = datetime.utcnow()
        self.duration_ms = int((self.end_time - self.start_time).total_seconds() * 1000)
        
        # Update span in trace context
        for span in self.trace_ctx["spans"]:
            if span["span_id"] == self.span_id:
                span["end_time"] = self.end_time.isoformat()
                span["duration_ms"] = self.duration_ms
                
                if exc_type is None:
                    span["status"] = "success"
                else:
                    span["status"] = "error"
                    span["error"] = {
                        "type": exc_type.__name__,
                        "message": str(exc_val)
                    }
                break
        
        return False  # Don't suppress exceptions
    
    def add_event(self, event: str, data: Optional[Dict[str, Any]] = None):
        """Add an event to the span"""
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "data": data or {}
        }
        
        for span in self.trace_ctx["spans"]:
            if span["span_id"] == self.span_id:
                if "events" not in span:
                    span["events"] = []
                span["events"].append(event_data)
                break


def get_trace_summary(trace_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of the trace
    
    Args:
        trace_ctx: Trace context
    
    Returns:
        Summary with timings and status
    """
    if not trace_ctx or "spans" not in trace_ctx:
        return {}
    
    spans = trace_ctx["spans"]
    
    # Calculate total time
    start_times = [span["start_time"] for span in spans if "start_time" in span]
    end_times = [span["end_time"] for span in spans if "end_time" in span]
    
    if not start_times or not end_times:
        total_duration = 0
    else:
        start = datetime.fromisoformat(min(start_times))
        end = datetime.fromisoformat(max(end_times))
        total_duration = int((end - start).total_seconds() * 1000)
    
    # Count status
    success_count = sum(1 for span in spans if span.get("status") == "success")
    error_count = sum(1 for span in spans if span.get("status") == "error")
    
    # Get span timings
    span_timings = {}
    for span in spans:
        if "duration_ms" in span:
            span_timings[span["name"]] = span["duration_ms"]
    
    return {
        "request_id": trace_ctx.get("request_id"),
        "trace_id": trace_ctx.get("trace_id"),
        "total_duration_ms": total_duration,
        "span_count": len(spans),
        "success_count": success_count,
        "error_count": error_count,
        "span_timings": span_timings,
        "spans": spans
    }


def format_trace_tree(trace_ctx: Dict[str, Any]) -> str:
    """
    Format trace as a tree for debugging
    
    Args:
        trace_ctx: Trace context
    
    Returns:
        Formatted tree string
    """
    if not trace_ctx or "spans" not in trace_ctx:
        return "No trace data"
    
    lines = []
    lines.append(f"Trace: {trace_ctx.get('trace_id', 'unknown')}")
    lines.append(f"Request: {trace_ctx.get('request_id', 'unknown')}")
    lines.append("")
    
    for i, span in enumerate(trace_ctx["spans"]):
        status_icon = "✅" if span.get("status") == "success" else "❌" if span.get("status") == "error" else "⏳"
        duration = span.get("duration_ms", "?")
        name = span.get("name", "unknown")
        
        lines.append(f"{status_icon} {name} ({duration}ms)")
        
        # Add events if present
        if "events" in span:
            for event in span["events"]:
                lines.append(f"    └─ {event['event']}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test tracing
    import time
    
    trace_ctx = create_trace_context("test-request-123")
    
    with Span(trace_ctx, "analyze_query", {"query_length": 50}):
        time.sleep(0.1)
    
    with Span(trace_ctx, "retrieve_docs") as span:
        time.sleep(0.2)
        span.add_event("fetched_legal", {"count": 10})
        span.add_event("fetched_gos", {"count": 5})
    
    with Span(trace_ctx, "synthesize_answer"):
        time.sleep(0.15)
    
    # Get summary
    summary = get_trace_summary(trace_ctx)
    print(f"Total duration: {summary['total_duration_ms']}ms")
    print(f"Spans: {summary['span_count']}")
    print("\nTrace tree:")
    print(format_trace_tree(trace_ctx))