const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const files = { '/': 'index.html', '/index.html': 'index.html', '/styles/style.css': 'styles/style.css', '/src/game/rules.js': 'src/game/rules.js', '/src/ui/game.js': 'src/ui/game.js' };
http.createServer((req, res) => {
  const file = files[new URL(req.url, 'http://localhost').pathname];
  if (!file) { res.writeHead(404); res.end(); return; }
  res.setHeader('Content-Type', ({ '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript' })[path.extname(file)]);
  res.end(fs.readFileSync(path.join(root, file)));
}).listen(4173, '127.0.0.1', () => console.log('http://127.0.0.1:4173'));
