// Authenticated manifests supply short-lived, episode-scoped worker URLs.
const routes = new Map();
export function setMediaRoute(manifest) {
  if (manifest.role === "qc" && manifest.media?.distributed)
    routes.set(manifest.id, manifest.media.media_base || null);
  else routes.delete(manifest.id);
}
export function mediaURL(session, suffix) {
  if (routes.has(session)) {
    const base = routes.get(session);
    if (!base) throw new Error("正在等待采集主机准备画面");
    return `${base}/${suffix}`;
  }
  return `/api/sessions/${session}/${suffix}`;
}
