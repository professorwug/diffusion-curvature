#!/usr/bin/env node
/**
 * build-index.js — Vercel build step for Zetteldev sites.
 *
 * Adds **verbose logging** so you can confirm the script is executed during
 * `vercel build` and see exactly which files are detected and emitted.
 *
 * 1. Recursively scans the provided source directory (default ".") for
 *    '*.html' (case‑insensitive) files **excluding** any index/404 we create.
 * 2. Emits three artefacts into the *output* directory:
 *      • index.html          – rich navigation UI (sidebar + iframe viewer)
 *      • index_fallback.html – simple unordered list of links
 *      • 404.html            – alias of index_fallback (Vercel shows on 404)
 *
 * Usage in package.json:
 *   "scripts": {
 *     "vercel-build": "node scripts/build-index.js"
 *   }
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// CLI arg 1 = source root; env BUILD_OUTPUT_DIR overrides output path.
const ROOT   = process.argv[2] || '.';
const OUTPUT = process.env.BUILD_OUTPUT_DIR || ROOT;

console.log('────────────────────────────────────────────────────────');
console.log('📦 Zetteldev build‑index running');
console.log(`• script dir  : ${__dirname}`);
console.log(`• CWD         : ${process.cwd()}`);
console.log(`• ROOT scan   : ${path.resolve(ROOT)}`);
console.log(`• OUTPUT dir  : ${path.resolve(OUTPUT)}`);
console.log('────────────────────────────────────────────────────────');

/** Recursively walk a directory and return paths to *.html (skip generated). */
async function findHtmlFiles(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const res = path.resolve(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await findHtmlFiles(res));
    } else if (/\.html?$/i.test(entry.name) && !/^index/i.test(entry.name) && !/^404\.html$/i.test(entry.name)) {
      files.push(path.relative(ROOT, res));
    }
  }
  return files.sort();
}

function generateRichIndex(files){
  const links = files.map(f=>{
    const label = path.basename(f, path.extname(f));
    return `<li><a href="#" onclick="load('${f}');return false;">${label}</a></li>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
  <title>Zetteldev Experiments</title>
  <style>
    body{margin:0;display:flex;height:100vh;font-family:system-ui,-apple-system,sans-serif;}
    nav{width:260px;padding:1rem;box-sizing:border-box;background:#f4f4f4;overflow-y:auto;border-right:1px solid #ddd;}
    nav h1{font-size:1.1rem;margin:0 0 .6rem 0;}
    nav ul{list-style:none;padding:0;}
    nav li{margin:.4rem 0;}
    nav a{text-decoration:none;color:#0366d6;}
    nav a:hover{text-decoration:underline;}
    iframe{flex:1 1 auto;border:none;width:100%;height:100%;}
  </style>
  <script>
    function load(url){const v=document.getElementById('viewer');v.src=url;document.title=url;history.replaceState(null,'',url);} 
    window.onload=()=>{const first='${files[0]??''}';if(first)load(first);}  
  </script>
</head>
<body>
  <nav>
    <h1>Experiments</h1>
    <ul>${links}</ul>
  </nav>
  <iframe id="viewer" title="Experiment viewer"></iframe>
</body></html>`;
}

function generateFallback(files){
  const items = files.map(f=>`<li><a href="${f}">${f}</a></li>`).join('\n');
  return `<!DOCTYPE html><meta charset='utf-8'><title>Zetteldev experiments</title><h1>Experiments</h1><ul>${items}</ul>`;
}

(async function main(){
  try{
    const htmlFiles = await findHtmlFiles(ROOT);
    console.log(`• HTML files found (${htmlFiles.length}):`);
    htmlFiles.forEach(f=>console.log('  –',f));

    const indexHtml    = generateRichIndex(htmlFiles);
    const fallbackHtml = generateFallback(htmlFiles);

    await fs.mkdir(OUTPUT, { recursive: true });
    await fs.writeFile(path.join(OUTPUT,'index.html'),           indexHtml);
    await fs.writeFile(path.join(OUTPUT,'index_fallback.html'),  fallbackHtml);
    await fs.writeFile(path.join(OUTPUT,'404.html'),             fallbackHtml);

    console.log('────────────────────────────────────────────────────────');
    console.log(`✅ Generated index.html + 404.html into ${OUTPUT}`);
    console.log('────────────────────────────────────────────────────────');
  }catch(err){
    console.error('💥 build-index failure:', err);
    process.exit(1);
  }
})();
