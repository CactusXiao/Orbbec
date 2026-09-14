// App shell only. No task API, result, access key or media is stored here.
const cacheName = "orbbec-browser-shell-v6";
const shell = [
  "/",
  "/app.js",
  "/auth.js",
  "/workflow.js",
  "/label-canvas.js",
  "/desktop-layout.js",
  "/queue.js",
  "/player.js",
  "/frame-cache.js",
  "/style.css",
];
self.addEventListener("install", (e) => {
  e.waitUntil(
    caches
      .open(cacheName)
      .then((c) => c.addAll(shell))
      .then(() => self.skipWaiting()),
  );
});
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter(
              (k) => k.startsWith("orbbec-browser-shell-") && k !== cacheName,
            )
            .map((k) => caches.delete(k)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (
    e.request.method !== "GET" ||
    u.origin !== location.origin ||
    !shell.includes(u.pathname)
  )
    return;
  e.respondWith(
    fetch(e.request)
      .then((r) => {
        if (r.ok) {
          const copy = r.clone();
          caches.open(cacheName).then((c) => c.put(e.request, copy));
        }
        return r;
      })
      .catch(() =>
        caches
          .open(cacheName)
          .then((c) => c.match(e.request))
          .then((r) => r || Response.error()),
      ),
  );
});
