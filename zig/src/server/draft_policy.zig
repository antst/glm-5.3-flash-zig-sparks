//! A request keeps its sampling and grammar when this engine can draft only plain greedy replies.
const api = @import("engine_api");
const json = @import("json");
const Cx = @import("errors.zig").Cx;

/// Default to ordinary target decoding; an explicit incompatible draft request is refused before admission.
pub fn prepare(info: api.Info, cx: *Cx, fields: json.Value, request: *api.Request) error{ Refused, OutOfMemory }!void {
    if (!info.greedy_unconstrained_drafts or !request.drafts or
        (request.sampling == null and request.structure == null)) return;
    if (fields.get("draft")) |asked| if (asked == .bool and asked.bool) {
        return cx.refuse("this serving configuration drafts only greedy requests without a token grammar; omit draft or set draft to false for this request");
    };
    request.drafts = false;
}
