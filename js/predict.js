document.addEventListener('DOMContentLoaded', function(){
  var form = document.getElementById('predictForm');
  var out = document.getElementById('predictResult');

  function refreshPlots(){
    var ts = Date.now();
    var imgs = [ ['confusionImg','confusion.png'], ['rocImg','roc.png'], ['decisionImg','decision.png'], ['sampleImg','sample_plot.png'] ];
    imgs.forEach(function(pair){
      try{
        var el = document.getElementById(pair[0]);
        if(!el) return;
        var url = 'http://127.0.0.1:5000/plots/' + pair[1] + '?t=' + ts;
        el.src = url;
        el.style.display = 'block';
        el.onerror = function(){ el.style.display = 'none'; };
      }catch(err){ }
    });
  }

  if(form){
    form.addEventListener('submit', function(e){
      e.preventDefault();
      out.textContent = 'Predicting...';
      var age = document.getElementById('inputAge').value;
      var salary = document.getElementById('inputSalary').value;
      fetch('http://127.0.0.1:5000/recommend', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({age: age, salary: salary})
      }).then(function(r){ return r.json(); })
      .then(function(j){
        if(j.error){ out.textContent = 'Error: ' + j.error; return; }
        out.innerHTML = '<strong>Sugerencia:</strong> ' + j.suggestion + '<br><strong>Probabilidad:</strong> ' + (j.probability*100).toFixed(2) + '%';
      }).catch(function(err){ out.textContent = 'Request failed: ' + err; });
    });
  }

  var uploadForm = document.getElementById('uploadForm');
  var uploadResult = document.getElementById('uploadResult');
  if(uploadForm){
    uploadForm.addEventListener('submit', function(e){
      e.preventDefault();
      if(!uploadResult) return;
      uploadResult.textContent = 'Uploading...';
      var modelFile = document.getElementById('inputModelFile').files[0];
      var scalerFile = document.getElementById('inputScalerFile').files[0];
      var fd = new FormData();
      if(modelFile) fd.append('model', modelFile);
      if(scalerFile) fd.append('scaler', scalerFile);
      fetch('http://127.0.0.1:5000/upload', { method: 'POST', body: fd })
      .then(function(r){ return r.json(); })
      .then(function(j){
        if(j.error){ uploadResult.textContent = 'Error: ' + j.error; return; }
        uploadResult.textContent = 'Saved: ' + (j.saved || []).join(', ');
        fetch('http://127.0.0.1:5000/generate_plots', { method: 'POST' })
        .then(function(r){ return r.json(); })
        .then(function(res){ if(!res.error) refreshPlots(); }).catch(function(){ });
      }).catch(function(err){ uploadResult.textContent = 'Upload failed: ' + err; });
    });
  }

  var reloadBtn = document.getElementById('reloadBtn');
  if(reloadBtn){
    reloadBtn.addEventListener('click', function(e){
      e.preventDefault();
      if(!uploadResult) return;
      uploadResult.textContent = 'Reloading...';
      fetch('http://127.0.0.1:5000/reload', { method: 'POST' })
      .then(function(r){ return r.json(); })
      .then(function(j){
        if(j.reloaded && j.reloaded.length>0) uploadResult.textContent = 'Reloaded: ' + j.reloaded.join(', ');
        else uploadResult.textContent = j.message || 'No changes detected';
        fetch('http://127.0.0.1:5000/generate_plots', { method: 'POST' })
        .then(function(r){ return r.json(); })
        .then(function(res){ if(!res.error) refreshPlots(); }).catch(function(){ });
      }).catch(function(err){ uploadResult.textContent = 'Reload failed: ' + err; });
    });
  }

  function populateFromServer(){
    var ageInput = document.getElementById('inputAge');
    var salInput = document.getElementById('inputSalary');
    if(ageInput) ageInput.value = '';
    if(salInput) salInput.value = '';
    var predictBtn = document.getElementById('predictBtn');
    if(predictBtn){ predictBtn.disabled = true; predictBtn.setAttribute('aria-disabled','true'); }

    fetch('http://127.0.0.1:5000/status').then(function(r){ if(!r.ok) throw new Error('status ' + r.status); return r.json(); })
    .then(function(s){
      var modelInfo = document.getElementById('modelInfo');
      if(modelInfo){
        var html = '';
        html += '<div style="color:#cfe8ff;margin-bottom:6px"><strong>Modelo cargado:</strong> ' + (s.model_loaded ? 'Sí' : 'No') + ' — <strong>Scaler cargado:</strong> ' + (s.scaler_loaded ? 'Sí' : 'No') + '</div>';
        if(s.model_loaded && s.scaler_loaded && s.scaler_mean){
          html += '<div style="color:#94a3b8">Valores por modelo: Edad = ' + (s.scaler_mean[0]).toFixed(2) + ', Salario = ' + (s.scaler_mean[1]).toFixed(2) + '</div>';
          if(ageInput) ageInput.value = Math.round(s.scaler_mean[0]);
          if(salInput) salInput.value = Math.round(s.scaler_mean[1]);
          if(predictBtn){ predictBtn.disabled = false; predictBtn.removeAttribute('aria-disabled'); }
        } else {
          html += '<div style="color:#94a3b8">El modelo o el scaler no están cargados. Los valores se mostrarán cuando el modelo esté disponible.</div>';
        }
        if(s.coef){ html += '<div style="color:#dfeff8;margin-top:6px"><strong>Coeficientes:</strong> ' + JSON.stringify(s.coef) + '</div>'; }
        modelInfo.innerHTML = html;
      }
    }).catch(function(err){ var modelInfo = document.getElementById('modelInfo'); if(modelInfo) modelInfo.innerHTML = '<div style="color:#ffb4b4">No se pudo obtener estado del servidor: ' + err + '</div>'; });

    fetch('http://127.0.0.1:5000/pipeline').then(function(r){ if(!r.ok) throw new Error('status ' + r.status); return r.json(); })
    .then(function(p){ var el = document.getElementById('pipelineInfo'); if(el && p && p.pipeline){ var html = '<div style="color:#94a3b8"><strong>Columnas extraídas:</strong> ' + (p.pipeline.extracted_columns || []).join(', ') + '</div>'; html += '<div style="margin-top:6px;color:#dfeff8"><strong>Pasos:</strong><ol style="color:#94a3b8">'; (p.pipeline.steps || []).forEach(function(s){ html += '<li>' + s + '</li>'; }); html += '</ol></div>'; el.innerHTML = html; } }).catch(function(){ });

    fetch('http://127.0.0.1:5000/metrics').then(function(r){ if(!r.ok) throw new Error('status ' + r.status); return r.json(); }).then(function(m){ var el = document.getElementById('metricsInfo'); if(el){ if(m && m.metrics && Object.keys(m.metrics).length>0){ var html = '<pre style="color:#dfeff8">' + JSON.stringify(m.metrics, null, 2) + '</pre>'; el.innerHTML = html; } else { el.innerHTML = '<div style="color:#94a3b8">No hay métricas exportadas desde Lab04. Puedes exportarlas a <code>model_metrics.json</code>.</div>'; } } }).catch(function(){ });

    fetch('http://127.0.0.1:5000/status').then(function(r){ if(!r.ok) throw new Error('status ' + r.status); return r.json(); }).then(function(s){
      var tbody = document.getElementById('datasetBody');
      if(!(s.model_loaded && s.scaler_loaded)){
        if(tbody) tbody.innerHTML = '<tr><td colspan="6" style="color:#94a3b8">Muestra no disponible: el modelo o el scaler no están cargados.</td></tr>';
        return;
      }
      fetch('http://127.0.0.1:5000/sample_rows?n=5').then(function(r){ if(!r.ok) throw new Error('status ' + r.status); return r.json(); }).then(function(d){ var tbody = document.getElementById('datasetBody'); if(tbody && d && Array.isArray(d.rows)){ tbody.innerHTML = ''; d.rows.forEach(function(row, idx){ var tr = document.createElement('tr'); var prob = ''; if(row['Probability']!==null && row['Probability']!==undefined && row['Probability'] !== ''){ var p = parseFloat(row['Probability']); if(p <= 1) prob = (p*100).toFixed(2) + '%'; else prob = p.toFixed(2) + '%'; } tr.innerHTML = '<th>' + idx + '</th>' + '<td>' + (row['User ID']||'') + '</td>' + '<td>' + (row['Gender']||'') + '</td>' + '<td>' + (typeof row['Age'] !== 'undefined' ? Number(row['Age']).toFixed(0) : '') + '</td>' + '<td>' + (typeof row['EstimatedSalary'] !== 'undefined' ? Number(row['EstimatedSalary']).toFixed(0) : '') + '</td>' + '<td>' + (row['Purchased']!==null && row['Purchased']!==undefined ? row['Purchased'] : '') + '</td>'; tbody.appendChild(tr); });
          // additionally, call /recommend for each sample row and show results in recommendationResult
          (function(rows){
            var recEl = document.getElementById('recommendationResult');
            if(!recEl) return;

            function renderResults(results){
              var out = '<pre style="color:#dfeff8;white-space:pre-wrap">';
              results.forEach(function(res){
                if(res.error){ out += (res.nombre||'') + ': ERROR - ' + res.error + '\n'; }
                else {
                  var pct = (res.probability*100).toFixed(1) + '%';
                  var decision = (res.prediction==1) ? 'COMPRARÁ' : 'NO COMPRARÁ';
                  out += (res.nombre||'') + ' — Edad: ' + (res.age||'') + ', Salario: $' + (res.salary||'') + ' ' + decision + ' (Confianza: ' + pct + ')\n';
                }
              });
              out += '</pre>';
              recEl.innerHTML = out;
            }

            // First try the batch endpoint; if it's missing or fails, fall back to per-row requests
            fetch('http://127.0.0.1:5000/recommend_batch', { method: 'GET' })
            .then(function(resp){
              if(resp.status === 404) throw { code: 404 };
              if(!resp.ok) throw new Error('status ' + resp.status);
              return resp.json();
            })
            .then(function(payload){
              if(payload && payload.results) renderResults(payload.results);
              else recEl.innerHTML = '<div style="color:#ffb4b4">Respuesta inesperada del servidor.</div>';
            })
            .catch(function(err){
              // fallback: do per-row calls to /recommend
              var fetches = rows.map(function(r){
                var age = Number(r['Age']) || 0;
                var sal = Number(r['EstimatedSalary']) || 0;
                return fetch('http://127.0.0.1:5000/recommend', {
                  method: 'POST',
                  headers: {'Content-Type':'application/json'},
                  body: JSON.stringify({age: age, salary: sal})
                }).then(function(resp){ return resp.json(); }).then(function(j){
                  if(j && !j.error) return { nombre: r.nombre || '', age: age, salary: sal, prediction: j.prediction, probability: j.probability, suggestion: j.suggestion };
                  return { nombre: r.nombre || '', error: j && j.error ? j.error : 'no response' };
                }).catch(function(e){ return { nombre: r.nombre || '', error: String(e) }; });
              });
              Promise.all(fetches).then(function(results){ renderResults(results); }).catch(function(e){ recEl.innerHTML = '<div style="color:#ffb4b4">No se pudieron obtener recomendaciones: '+e+'</div>'; });
            });
          })(d.rows);
        } }).catch(function(err){ var tbody = document.getElementById('datasetBody'); if(tbody) tbody.innerHTML = '<tr><td colspan="6" style="color:#ffb4b4">No se pudieron cargar filas de muestra: ' + err + '</td></tr>'; });
    }).catch(function(err){ var tbody = document.getElementById('datasetBody'); if(tbody) tbody.innerHTML = '<tr><td colspan="6" style="color:#ffb4b4">No se pudieron cargar filas de muestra: ' + err + '</td></tr>'; });
  }

  populateFromServer();
});
document.addEventListener('DOMContentLoaded', function(){
  var form = document.getElementById('predictForm');
  var out = document.getElementById('predictResult');

  function refreshPlots(){
    var ts = Date.now();
    var imgs = [ ['confusionImg','confusion.png'], ['rocImg','roc.png'], ['decisionImg','decision.png'], ['sampleImg','sample_plot.png'] ];
    imgs.forEach(function(pair){
      try{
        var el = document.getElementById(pair[0]);
        if(!el) return;
        // Use absolute URL to ensure requests go to the Flask server on port 5000
        var url = 'http://127.0.0.1:5000/plots/' + pair[1] + '?t=' + ts;
        el.src = url;
        el.style.display = 'block';
        // hide image if it fails to load
        el.onerror = function(){ el.style.display = 'none'; };
      }catch(err){ /* ignore per-image errors */ }
    });
  }

  if(form){
    form.addEventListener('submit', function(e){
      e.preventDefault();
      out.textContent = 'Predicting...';
      var age = document.getElementById('inputAge').value;
      var salary = document.getElementById('inputSalary').value;
      fetch('http://127.0.0.1:5000/recommend', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({age: age, salary: salary})
      }).then(function(r){ return r.json(); })
      .then(function(j){
        if(j.error){ out.textContent = 'Error: ' + j.error; return; }
        out.innerHTML = '<strong>Sugerencia:</strong> ' + j.suggestion + '<br><strong>Probabilidad:</strong> ' + (j.probability*100).toFixed(2) + '%';
      }).catch(function(err){ out.textContent = 'Request failed: ' + err; });
    });
  }

  // Upload form (model + scaler)
  var uploadForm = document.getElementById('uploadForm');
  var uploadResult = document.getElementById('uploadResult');
  if(uploadForm){
    uploadForm.addEventListener('submit', function(e){
      e.preventDefault();
      if(!uploadResult) return;
      uploadResult.textContent = 'Uploading...';
      var modelFile = document.getElementById('inputModelFile').files[0];
      var scalerFile = document.getElementById('inputScalerFile').files[0];
      var fd = new FormData();
      if(modelFile) fd.append('model', modelFile);
      if(scalerFile) fd.append('scaler', scalerFile);
      fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: fd
      }).then(function(r){ return r.json(); })
      .then(function(j){
        if(j.error){ uploadResult.textContent = 'Error: ' + j.error; return; }
        uploadResult.textContent = 'Saved: ' + (j.saved || []).join(', ');
        // Trigger plot generation and refresh images
        fetch('http://127.0.0.1:5000/generate_plots', { method: 'POST' })
        .then(function(r){ return r.json(); })
        .then(function(res){ if(!res.error) refreshPlots(); }).catch(function(){ /* ignore */ });
      }).catch(function(err){ uploadResult.textContent = 'Upload failed: ' + err; });
    });
  }

  // Reload button: POST /reload
  var reloadBtn = document.getElementById('reloadBtn');
  if(reloadBtn){
    reloadBtn.addEventListener('click', function(e){
      e.preventDefault();
      if(!uploadResult) return;
      uploadResult.textContent = 'Reloading...';
      fetch('http://127.0.0.1:5000/reload', { method: 'POST' })
      .then(function(r){ return r.json(); })
      .then(function(j){
        if(j.reloaded && j.reloaded.length>0) uploadResult.textContent = 'Reloaded: ' + j.reloaded.join(', ');
        else uploadResult.textContent = j.message || 'No changes detected';
        // After reload, regenerate plots
        fetch('http://127.0.0.1:5000/generate_plots', { method: 'POST' })
        .then(function(r){ return r.json(); })
        .then(function(res){ if(!res.error) refreshPlots(); }).catch(function(){ /* ignore */ });
      }).catch(function(err){ uploadResult.textContent = 'Reload failed: ' + err; });
    });
  }

  // On load: get status and sample to populate the page automatically
  function populateFromServer(){
    // status -> populate inputs and model info
    // clear inputs first; only set them from the server when model+scaler are loaded
    var ageInput = document.getElementById('inputAge');
    var salInput = document.getElementById('inputSalary');
    if(ageInput) ageInput.value = '';
    if(salInput) salInput.value = '';
    var predictBtn = document.getElementById('predictBtn');
    if(predictBtn){ predictBtn.disabled = true; predictBtn.setAttribute('aria-disabled','true'); }

    fetch('http://127.0.0.1:5000/status').then(function(r){
      if(!r.ok) throw new Error('status ' + r.status);
      return r.json();
    }).then(function(s){
      var modelInfo = document.getElementById('modelInfo');
      if(modelInfo){
        var html = '';
        html += '<div style="color:#cfe8ff;margin-bottom:6px"><strong>Modelo cargado:</strong> ' + (s.model_loaded ? 'Sí' : 'No') + ' — <strong>Scaler cargado:</strong> ' + (s.scaler_loaded ? 'Sí' : 'No') + '</div>';
        if(s.model_loaded && s.scaler_loaded && s.scaler_mean){
          // only populate inputs from the model/scaler values
          html += '<div style="color:#94a3b8">Valores por modelo: Edad = ' + (s.scaler_mean[0]).toFixed(2) + ', Salario = ' + (s.scaler_mean[1]).toFixed(2) + '</div>';
          if(ageInput) ageInput.value = Math.round(s.scaler_mean[0]);
          if(salInput) salInput.value = Math.round(s.scaler_mean[1]);
          if(predictBtn){ predictBtn.disabled = false; predictBtn.removeAttribute('aria-disabled'); }
        } else {
          html += '<div style="color:#94a3b8">El modelo o el scaler no están cargados. Los valores se mostrarán cuando el modelo esté disponible.</div>';
        }
        if(s.coef){
          html += '<div style="color:#dfeff8;margin-top:6px"><strong>Coeficientes:</strong> ' + JSON.stringify(s.coef) + '</div>';
        }
        modelInfo.innerHTML = html;
      }
    }).catch(function(err){
      var modelInfo = document.getElementById('modelInfo');
      if(modelInfo) modelInfo.innerHTML = '<div style="color:#ffb4b4">No se pudo obtener estado del servidor: ' + err + '</div>';
    });

    // pipeline -> populate pipelineInfo
    fetch('http://127.0.0.1:5000/pipeline').then(function(r){
      if(!r.ok) throw new Error('status ' + r.status);
      return r.json();
    }).then(function(p){
      var el = document.getElementById('pipelineInfo');
      if(el && p && p.pipeline){
        var html = '<div style="color:#94a3b8"><strong>Columnas extraídas:</strong> ' + (p.pipeline.extracted_columns || []).join(', ') + '</div>';
        html += '<div style="margin-top:6px;color:#dfeff8"><strong>Pasos:</strong><ol style="color:#94a3b8">';
        (p.pipeline.steps || []).forEach(function(s){ html += '<li>' + s + '</li>'; });
        html += '</ol></div>';
        el.innerHTML = html;
      }
    }).catch(function(){ /* ignore */ });

    // metrics -> populate metricsInfo
    fetch('http://127.0.0.1:5000/metrics').then(function(r){
      if(!r.ok) throw new Error('status ' + r.status);
      return r.json();
    }).then(function(m){
      var el = document.getElementById('metricsInfo');
      if(el){
        if(m && m.metrics && Object.keys(m.metrics).length>0){
          var html = '<pre style="color:#dfeff8">' + JSON.stringify(m.metrics, null, 2) + '</pre>';
          el.innerHTML = html;
        } else {
          el.innerHTML = '<div style="color:#94a3b8">No hay métricas exportadas desde Lab04. Puedes exportarlas a <code>model_metrics.json</code>.</div>';
        }
      }
    }).catch(function(){ /* ignore */ });

    // sample rows -> populate dataset table (fetch up to 5 rows)
    // Only request sample rows if model+scaler are loaded. Otherwise leave table empty.
    fetch('http://127.0.0.1:5000/status').then(function(r){
      if(!r.ok) throw new Error('status ' + r.status);
      return r.json();
    }).then(function(s){
      var tbody = document.getElementById('datasetBody');
      if(!(s.model_loaded && s.scaler_loaded)){
        if(tbody) tbody.innerHTML = '<tr><td colspan="6" style="color:#94a3b8">Muestra no disponible: el modelo o el scaler no están cargados.</td></tr>';
        return;
      }
      // model+scaler are loaded -> fetch sample rows
      fetch('http://127.0.0.1:5000/sample_rows?n=5').then(function(r){
        if(!r.ok) throw new Error('status ' + r.status);
        return r.json();
      }).then(function(d){
        var tbody = document.getElementById('datasetBody');
        if(tbody && d && Array.isArray(d.rows)){
          tbody.innerHTML = '';
          d.rows.forEach(function(row, idx){
            var tr = document.createElement('tr');
            var prob = '';
            if(row['Probability']!==null && row['Probability']!==undefined && row['Probability'] !== ''){
              var p = parseFloat(row['Probability']);
              if(p <= 1) prob = (p*100).toFixed(2) + '%'; else prob = p.toFixed(2) + '%';
            }
            tr.innerHTML = '<th>' + idx + '</th>' +
              '<td>' + (row['User ID']||'') + '</td>' +
              '<td>' + (row['Gender']||'') + '</td>' +
              '<td>' + (typeof row['Age'] !== 'undefined' ? Number(row['Age']).toFixed(0) : '') + '</td>' +
              '<td>' + (typeof row['EstimatedSalary'] !== 'undefined' ? Number(row['EstimatedSalary']).toFixed(0) : '') + '</td>' +
              '<td>' + (row['Purchased']!==null && row['Purchased']!==undefined ? row['Purchased'] : '') + '</td>';
            tbody.appendChild(tr);
          });
        }
      }).catch(function(err){
        var tbody = document.getElementById('datasetBody');
        if(tbody) tbody.innerHTML = '<tr><td colspan="6" style="color:#ffb4b4">No se pudieron cargar filas de muestra: ' + err + '</td></tr>';
      });
    }).catch(function(err){
      var tbody = document.getElementById('datasetBody');
      if(tbody) tbody.innerHTML = '<tr><td colspan="6" style="color:#ffb4b4">No se pudieron cargar filas de muestra: ' + err + '</td></tr>';
    });
  }

  // call populate on ready
  populateFromServer();
});
document.addEventListener('DOMContentLoaded', function(){
  var form = document.getElementById('predictForm');
  var out = document.getElementById('predictResult');
  if(form){
    form.addEventListener('submit', function(e){
    e.preventDefault();
    out.textContent = 'Predicting...';
    var age = document.getElementById('inputAge').value;
    var salary = document.getElementById('inputSalary').value;
    fetch('http://127.0.0.1:5000/recommend', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({age: age, salary: salary})
    }).then(function(r){ return r.json(); })
    .then(function(j){
      if(j.error){ out.textContent = 'Error: ' + j.error; return; }
      out.innerHTML = '<strong>Sugerencia:</strong> ' + j.suggestion + '<br><strong>Probabilidad:</strong> ' + (j.probability*100).toFixed(2) + '%';
    }).catch(function(err){ out.textContent = 'Request failed: ' + err; });
    });
  }

  // Upload form (model + scaler)
  var uploadForm = document.getElementById('uploadForm');
  var uploadResult = document.getElementById('uploadResult');
  if(uploadForm){
    uploadForm.addEventListener('submit', function(e){
      e.preventDefault();
      if(!uploadResult) return;
      uploadResult.textContent = 'Uploading...';
      var modelFile = document.getElementById('inputModelFile').files[0];
      var scalerFile = document.getElementById('inputScalerFile').files[0];
      var fd = new FormData();
      if(modelFile) fd.append('model', modelFile);
      if(scalerFile) fd.append('scaler', scalerFile);
      fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: fd
      }).then(function(r){ return r.json(); })
      .then(function(j){
        if(j.error){ uploadResult.textContent = 'Error: ' + j.error; return; }
        uploadResult.textContent = 'Saved: ' + (j.saved || []).join(', ');
      }).catch(function(err){ uploadResult.textContent = 'Upload failed: ' + err; });
    });
  }

  // Reload button: POST /reload
  var reloadBtn = document.getElementById('reloadBtn');
  if(reloadBtn){
    reloadBtn.addEventListener('click', function(e){
      e.preventDefault();
      if(!uploadResult) return;
      uploadResult.textContent = 'Reloading...';
      fetch('http://127.0.0.1:5000/reload', { method: 'POST' })
      .then(function(r){ return r.json(); })
      .then(function(j){
        if(j.reloaded && j.reloaded.length>0) uploadResult.textContent = 'Reloaded: ' + j.reloaded.join(', ');
        else uploadResult.textContent = j.message || 'No changes detected';
      }).catch(function(err){ uploadResult.textContent = 'Reload failed: ' + err; });
    });
  }

  // On load: get status and sample to populate the page automatically
  function populateFromServer(){
  // status -> populate inputs and model info
    // clear inputs first; only set them from the server when model+scaler are loaded
    var ageInput = document.getElementById('inputAge');
    var salInput = document.getElementById('inputSalary');
    if(ageInput) ageInput.value = '';
    if(salInput) salInput.value = '';
    var predictBtn = document.getElementById('predictBtn');
    if(predictBtn){ predictBtn.disabled = true; predictBtn.setAttribute('aria-disabled','true'); }

    fetch('http://127.0.0.1:5000/status').then(function(r){
      if(!r.ok) throw new Error('status ' + r.status);
      return r.json();
    }).then(function(s){
      var modelInfo = document.getElementById('modelInfo');
      if(modelInfo){
        var html = '';
        html += '<div style="color:#cfe8ff;margin-bottom:6px"><strong>Modelo cargado:</strong> ' + (s.model_loaded ? 'Sí' : 'No') + ' — <strong>Scaler cargado:</strong> ' + (s.scaler_loaded ? 'Sí' : 'No') + '</div>';
        if(s.model_loaded && s.scaler_loaded && s.scaler_mean){
          // only populate inputs from the model/scaler values
          html += '<div style="color:#94a3b8">Valores por modelo: Edad = ' + (s.scaler_mean[0]).toFixed(2) + ', Salario = ' + (s.scaler_mean[1]).toFixed(2) + '</div>';
          if(ageInput) ageInput.value = Math.round(s.scaler_mean[0]);
          if(salInput) salInput.value = Math.round(s.scaler_mean[1]);
          if(predictBtn){ predictBtn.disabled = false; predictBtn.removeAttribute('aria-disabled'); }
        } else {
          html += '<div style="color:#94a3b8">El modelo o el scaler no están cargados. Los valores se mostrarán cuando el modelo esté disponible.</div>';
        }
        if(s.coef){
          html += '<div style="color:#dfeff8;margin-top:6px"><strong>Coeficientes:</strong> ' + JSON.stringify(s.coef) + '</div>';
        }
        modelInfo.innerHTML = html;
      }
    }).catch(function(err){
      var modelInfo = document.getElementById('modelInfo');
      if(modelInfo) modelInfo.innerHTML = '<div style="color:#ffb4b4">No se pudo obtener estado del servidor: ' + err + '</div>';
    });

    // pipeline -> populate pipelineInfo
    fetch('http://127.0.0.1:5000/pipeline').then(function(r){
      if(!r.ok) throw new Error('status ' + r.status);
      return r.json();
    }).then(function(p){
      var el = document.getElementById('pipelineInfo');
      if(el && p && p.pipeline){
        var html = '<div style="color:#94a3b8"><strong>Columnas extraídas:</strong> ' + (p.pipeline.extracted_columns || []).join(', ') + '</div>';
        html += '<div style="margin-top:6px;color:#dfeff8"><strong>Pasos:</strong><ol style="color:#94a3b8">';
        (p.pipeline.steps || []).forEach(function(s){ html += '<li>' + s + '</li>'; });
        html += '</ol></div>';
        el.innerHTML = html;
      }
    }).catch(function(){ /* ignore */ });

    // metrics -> populate metricsInfo
    fetch('http://127.0.0.1:5000/metrics').then(function(r){
      if(!r.ok) throw new Error('status ' + r.status);
      return r.json();
    }).then(function(m){
      var el = document.getElementById('metricsInfo');
      if(el){
        if(m && m.metrics && Object.keys(m.metrics).length>0){
          var html = '<pre style="color:#dfeff8">' + JSON.stringify(m.metrics, null, 2) + '</pre>';
          el.innerHTML = html;
        } else {
          el.innerHTML = '<div style="color:#94a3b8">No hay métricas exportadas desde Lab04. Puedes exportarlas a <code>model_metrics.json</code>.</div>';
        }
      }
    }).catch(function(){ /* ignore */ });

    // sample rows -> populate dataset table (fetch up to 5 rows)
    // Only request sample rows if model+scaler are loaded. Otherwise leave table empty.
    fetch('http://127.0.0.1:5000/status').then(function(r){
      if(!r.ok) throw new Error('status ' + r.status);
      return r.json();
    }).then(function(s){
      var tbody = document.getElementById('datasetBody');
      if(!(s.model_loaded && s.scaler_loaded)){
        if(tbody) tbody.innerHTML = '<tr><td colspan="6" style="color:#94a3b8">Muestra no disponible: el modelo o el scaler no están cargados.</td></tr>';
        return;
      }
      // model+scaler are loaded -> fetch sample rows
      fetch('http://127.0.0.1:5000/sample_rows?n=5').then(function(r){
        if(!r.ok) throw new Error('status ' + r.status);
        return r.json();
      }).then(function(d){
        var tbody = document.getElementById('datasetBody');
        if(tbody && d && Array.isArray(d.rows)){
          tbody.innerHTML = '';
          d.rows.forEach(function(row, idx){
            var tr = document.createElement('tr');
            var prob = '';
            if(row['Probability']!==null && row['Probability']!==undefined && row['Probability'] !== ''){
              var p = parseFloat(row['Probability']);
              if(p <= 1) prob = (p*100).toFixed(2) + '%'; else prob = p.toFixed(2) + '%';
            }
            tr.innerHTML = '<th>' + idx + '</th>' +
              '<td>' + (row['User ID']||'') + '</td>' +
              '<td>' + (row['Gender']||'') + '</td>' +
              '<td>' + (typeof row['Age'] !== 'undefined' ? Number(row['Age']).toFixed(0) : '') + '</td>' +
              '<td>' + (typeof row['EstimatedSalary'] !== 'undefined' ? Number(row['EstimatedSalary']).toFixed(0) : '') + '</td>' +
              '<td>' + (row['Purchased']!==null && row['Purchased']!==undefined ? row['Purchased'] : '') + '</td>';
            tbody.appendChild(tr);
          });
        }
      }).catch(function(err){
        var tbody = document.getElementById('datasetBody');
        if(tbody) tbody.innerHTML = '<tr><td colspan="6" style="color:#ffb4b4">No se pudieron cargar filas de muestra: ' + err + '</td></tr>';
      });
    }).catch(function(err){
      var tbody = document.getElementById('datasetBody');
      if(tbody) tbody.innerHTML = '<tr><td colspan="6" style="color:#ffb4b4">No se pudieron cargar filas de muestra: ' + err + '</td></tr>';
    });
  }

  // call populate on ready
  populateFromServer();
});
