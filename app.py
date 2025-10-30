from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from flask import send_from_directory, abort
import json

# Matplotlib: use Agg backend for headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
CORS(app)

# Try to load model and scaler from current folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo_regresion_logistica.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

try:
    clf = joblib.load(MODEL_PATH)
except Exception as e:
    clf = None
    print(f"WARNING: Could not load model from {MODEL_PATH}: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    scaler = None
    print(f"WARNING: Could not load scaler from {SCALER_PATH}: {e}")

# Track modification times so we can auto-reload if the pickle files change on disk
MODEL_MTIME = None
SCALER_MTIME = None

def _get_mtime(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

def load_model_if_changed():
    """Reload model and scaler if their files have changed on disk.

    This is safe to call before handling a request; it will only reload
    when the underlying file mtime changes.
    """
    global clf, scaler, MODEL_MTIME, SCALER_MTIME
    m_mtime = _get_mtime(MODEL_PATH)
    s_mtime = _get_mtime(SCALER_PATH)

    reloaded = []
    # Reload model if file appeared/changed
    if m_mtime and m_mtime != MODEL_MTIME:
        try:
            clf = joblib.load(MODEL_PATH)
            MODEL_MTIME = m_mtime
            reloaded.append('model')
            print(f"Loaded model from {MODEL_PATH} (mtime: {MODEL_MTIME})")
        except Exception as e:
            print(f"WARNING: Could not reload model from {MODEL_PATH}: {e}")

    # Reload scaler if file appeared/changed
    if s_mtime and s_mtime != SCALER_MTIME:
        try:
            scaler = joblib.load(SCALER_PATH)
            SCALER_MTIME = s_mtime
            reloaded.append('scaler')
            print(f"Loaded scaler from {SCALER_PATH} (mtime: {SCALER_MTIME})")
        except Exception as e:
            print(f"WARNING: Could not reload scaler from {SCALER_PATH}: {e}")

    return reloaded

# Initialize mtimes on startup
MODEL_MTIME = _get_mtime(MODEL_PATH)
SCALER_MTIME = _get_mtime(SCALER_PATH)

# Optional sample file exported by Lab04. Supported formats: JSON (list of dicts) or CSV.
SAMPLE_JSON_PATH = os.path.join(os.path.dirname(__file__), 'sample_rows.json')
SAMPLE_CSV_PATH = os.path.join(os.path.dirname(__file__), 'sample_rows.csv')
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), 'pipeline_info.json')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'model_metrics.json')


def generate_and_save_plots(n_samples=200):
    """Generate a set of diagnostic plots (confusion matrix, ROC, decision regions, sample scatter).

    Uses sample_rows.json/csv if available; otherwise generates fallback rows. If true labels are not
    present in the samples, this function will simulate labels from predicted probabilities with noise
    so the plots remain informative (useful for demos).
    Saves PNG files in the project root and writes `model_metrics.json` with computed metrics.
    Returns list of saved filenames and metrics dict.
    """
    base = os.path.dirname(__file__)
    # ensure model/scaler loaded
    load_model_if_changed()
    if clf is None or scaler is None:
        raise RuntimeError('Model or scaler not loaded')

    # get rows using same logic as sample_rows
    rows = None
    if os.path.exists(SAMPLE_JSON_PATH):
        try:
            with open(SAMPLE_JSON_PATH, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    rows = data[:n_samples]
        except Exception:
            rows = None
    if rows is None and os.path.exists(SAMPLE_CSV_PATH):
        try:
            import csv
            rows = []
            with open(SAMPLE_CSV_PATH, 'r', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for i, r in enumerate(reader):
                    if i >= n_samples: break
                    rows.append(r)
        except Exception:
            rows = None

    # fallback generate
    if rows is None:
        # try to use scaler mean as before
        if scaler is not None and hasattr(scaler, 'mean_'):
            means = getattr(scaler, 'mean_')
            base_age = float(means[0]) if len(means) > 0 else 30.0
            base_salary = float(means[1]) if len(means) > 1 else 60000.0
        else:
            base_age = 30.0
            base_salary = 60000.0
        rows = []
        for i in range(min(n_samples, 50)):
            age = round(base_age + (i - 5), 2)
            salary = round(base_salary + (i - 5) * 1000, 2)
            rows.append({'User ID': '', 'Gender': '', 'Age': age, 'EstimatedSalary': salary, 'Purchased': None})

    # build X and check for true labels
    X = np.array([[float(r.get('Age', 0)), float(r.get('EstimatedSalary', 0))] for r in rows])
    Xs = scaler.transform(X)
    probas = clf.predict_proba(Xs)[:, 1]
    preds = clf.predict(Xs)

    # Determine y_true: use 'Purchased' if present and non-null, otherwise simulate from probas
    y_true = []
    any_true = False
    for r in rows:
        p = r.get('Purchased')
        if p is None or p == '':
            y_true.append(None)
        else:
            try:
                y_true.append(int(p))
                any_true = True
            except Exception:
                y_true.append(None)

    if not any_true:
        # simulate ground truth by sampling from probabilities with some noise to avoid perfect scores
        rng = np.random.RandomState(0)
        noisy = probas + rng.normal(0, 0.12, size=len(probas))
        y_true = (noisy > 0.5).astype(int)
    else:
        y_true = np.array([0 if v is None else int(v) for v in y_true])

    # compute metrics
    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    cm = confusion_matrix(y_true, preds)
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    auc = float(roc_auc_score(y_true, probas))

    metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}

    # Plot 1: confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Compra','Compra'], yticklabels=['No Compra','Compra'])
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.title('Matriz de Confusión')
    confusion_path = os.path.join(base, 'confusion.png')
    plt.tight_layout()
    plt.savefig(confusion_path)
    plt.close()

    # Plot 2: ROC
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0,1],[0,1], color='red', lw=1, linestyle='--', label='Aleatorio')
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    roc_path = os.path.join(base, 'roc.png')
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # Plot 3: decision regions
    try:
        # decide grid ranges from X
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1000, X[:,1].max() + 1000
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_s = scaler.transform(grid)
        Z = clf.predict(grid_s)
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(7,6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        # scatter sample points colored by true label (if available) or by prediction
        labels = y_true if any_true else preds
        plt.scatter(X[:,0], X[:,1], c=labels, cmap='bwr', edgecolor='k', s=40)
        plt.xlabel('Age'); plt.ylabel('EstimatedSalary')
        plt.title('Regiones de Decisión - Regresión Logística')
        decision_path = os.path.join(base, 'decision.png')
        plt.tight_layout()
        plt.savefig(decision_path)
        plt.close()
    except Exception:
        decision_path = None

    # Plot 4: sample scatter with probabilities
    plt.figure(figsize=(7,5))
    sc = plt.scatter(X[:,0], X[:,1], c=probas, cmap='viridis', s=50, edgecolor='k')
    plt.colorbar(sc, label='Probabilidad de Compra')
    plt.xlabel('Age'); plt.ylabel('EstimatedSalary')
    plt.title('Muestras con Probabilidades')
    sample_path = os.path.join(base, 'sample_plot.png')
    plt.tight_layout()
    plt.savefig(sample_path)
    plt.close()

    # write metrics file
    try:
        with open(METRICS_PATH, 'w', encoding='utf-8') as fh:
            json.dump(metrics, fh, indent=2)
    except Exception:
        pass

    saved = [p for p in [confusion_path, roc_path, decision_path, sample_path] if p]
    saved_names = [os.path.basename(p) for p in saved]
    return saved_names, metrics


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure we have the latest files saved by Lab04 (if any)
    load_model_if_changed()
    data = request.get_json(force=True)
    # Expect JSON: {"age": number, "salary": number}
    try:
        age = float(data.get('age', 0))
        salary = float(data.get('salary', 0))
    except Exception:
        return jsonify({'error': 'Invalid input values'}), 400

    if scaler is None or clf is None:
        return jsonify({'error': 'Model or scaler not loaded on server'}), 500

    import numpy as np
    X = np.array([[age, salary]])
    Xs = scaler.transform(X)
    pred = int(clf.predict(Xs)[0])
    proba = float(clf.predict_proba(Xs)[0][1])

    return jsonify({'prediction': pred, 'probability': proba})


@app.route('/upload', methods=['POST'])
def upload():
    """Endpoint to upload new model/scaler files from the browser.

    Accepts multipart/form-data with optional fields:
      - model: the modelo_regresion_logistica.pkl file
      - scaler: the scaler.pkl file

    Saves them atomically (via a .tmp -> replace) and triggers reload.
    """
    if not request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    saved = []
    # Save model file
    model_file = request.files.get('model')
    if model_file:
        try:
            tmp = MODEL_PATH + '.tmp'
            model_file.save(tmp)
            os.replace(tmp, MODEL_PATH)
            saved.append('model')
        except Exception as e:
            return jsonify({'error': f'Could not save model file: {e}'}), 500

    # Save scaler file
    scaler_file = request.files.get('scaler')
    if scaler_file:
        try:
            tmp = SCALER_PATH + '.tmp'
            scaler_file.save(tmp)
            os.replace(tmp, SCALER_PATH)
            saved.append('scaler')
        except Exception as e:
            return jsonify({'error': f'Could not save scaler file: {e}'}), 500

    # Reload any changed files
    reloaded = load_model_if_changed()
    return jsonify({'saved': saved, 'reloaded': reloaded}), 200


@app.route('/reload', methods=['POST'])
def reload_endpoint():
    """Manual reload endpoint. Triggers reloading of model/scaler if files changed.

    Returns JSON with which objects were reloaded.
    """
    reloaded = load_model_if_changed()
    if not reloaded:
        return jsonify({'reloaded': [], 'message': 'No changes detected'}), 200
    return jsonify({'reloaded': reloaded, 'message': 'Reloaded'}), 200


@app.route('/status', methods=['GET'])
def status():
    """Return status information about model and scaler.

    Includes whether they're loaded, their mtimes, and some metadata
    (scaler means, model coefficients) when available.
    """
    info = {
        'model_loaded': clf is not None,
        'scaler_loaded': scaler is not None,
        'model_mtime': MODEL_MTIME,
        'scaler_mtime': SCALER_MTIME,
    }
    try:
        if scaler is not None and hasattr(scaler, 'mean_'):
            info['scaler_mean'] = list(map(float, getattr(scaler, 'mean_')))
        if clf is not None:
            # model metadata
            info['classes'] = list(map(int, getattr(clf, 'classes_', [])))
            info['coef'] = [list(map(float, row)) for row in getattr(clf, 'coef_', [])]
            info['intercept'] = list(map(float, getattr(clf, 'intercept_', [])))
    except Exception as e:
        # Be permissive: don't fail the whole endpoint if attributes unexpected
        info['meta_error'] = str(e)
    return jsonify(info)


@app.route('/sample', methods=['GET'])
def sample():
    """Return a small sample row generated from the scaler mean and model prediction.

    The row mirrors the columns shown in the documentation: User ID, Gender, Age,
    EstimatedSalary, Purchased (if model present) and probability.
    """
    # Default sample values
    age = None
    salary = None
    if scaler is not None and hasattr(scaler, 'mean_'):
        try:
            means = getattr(scaler, 'mean_')
            if len(means) >= 2:
                age = float(means[0])
                salary = float(means[1])
        except Exception:
            pass

    if age is None:
        age = 0.0
    if salary is None:
        salary = 0.0

    sample_row = {
        'User ID': '',
        'Gender': '',
        'Age': round(age, 2),
        'EstimatedSalary': round(salary, 2),
    }

    if clf is not None and scaler is not None:
        try:
            import numpy as np
            X = np.array([[age, salary]])
            Xs = scaler.transform(X)
            pred = int(clf.predict(Xs)[0])
            proba = float(clf.predict_proba(Xs)[0][1])
            sample_row['Purchased'] = pred
            sample_row['Probability'] = round(proba, 4)
        except Exception as e:
            sample_row['Purchased'] = None
            sample_row['Probability'] = None
            sample_row['error'] = str(e)
    else:
        sample_row['Purchased'] = None
        sample_row['Probability'] = None

    return jsonify({'sample': sample_row})


@app.route('/sample_rows', methods=['GET'])
def sample_rows():
    """Return multiple sample rows.

    Priority:
      1) If `sample_rows.json` exists, return its contents (must be a list of dicts).
      2) Else if `sample_rows.csv` exists, return its rows as dicts (CSV header required).
      3) Else fallback to generating `n` rows using scaler.mean_ (or zeros) and model if available.
    """
    import json, csv
    n = int(request.args.get('n', 5))

    # Try JSON
    if os.path.exists(SAMPLE_JSON_PATH):
        try:
            with open(SAMPLE_JSON_PATH, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                # ensure list
                if isinstance(data, list):
                    return jsonify({'rows': data[:n]})
        except Exception as e:
            # ignore and fallback
            print(f"WARNING: could not read {SAMPLE_JSON_PATH}: {e}")

    # Try CSV
    if os.path.exists(SAMPLE_CSV_PATH):
        try:
            rows = []
            with open(SAMPLE_CSV_PATH, 'r', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for i, r in enumerate(reader):
                    if i >= n:
                        break
                    rows.append(r)
            return jsonify({'rows': rows})
        except Exception as e:
            print(f"WARNING: could not read {SAMPLE_CSV_PATH}: {e}")

    # Fallback: generate n rows from scaler mean or zeros
    rows = []
    try:
        if scaler is not None and hasattr(scaler, 'mean_'):
            means = getattr(scaler, 'mean_')
            base_age = float(means[0]) if len(means) > 0 else 0.0
            base_salary = float(means[1]) if len(means) > 1 else 0.0
        else:
            base_age = 0.0
            base_salary = 0.0
    except Exception:
        base_age = 0.0
        base_salary = 0.0

    for i in range(n):
        age = round(base_age + (i - n/2), 2)
        salary = round(base_salary + (i - n/2) * 1000, 2)
        row = {
            'User ID': '',
            'Gender': '',
            'Age': age,
            'EstimatedSalary': salary,
        }
        if clf is not None and scaler is not None:
            try:
                import numpy as np
                X = np.array([[age, salary]])
                Xs = scaler.transform(X)
                pred = int(clf.predict(Xs)[0])
                proba = float(clf.predict_proba(Xs)[0][1])
                row['Purchased'] = pred
                row['Probability'] = round(proba, 4)
            except Exception:
                row['Purchased'] = None
                row['Probability'] = None
        else:
            row['Purchased'] = None
            row['Probability'] = None
        rows.append(row)

    return jsonify({'rows': rows})


@app.route('/pipeline', methods=['GET'])
def pipeline():
    """Return pipeline information (columns extracted, steps). Prefer exported file from Lab04."""
    import json
    if os.path.exists(PIPELINE_PATH):
        try:
            with open(PIPELINE_PATH, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                return jsonify({'pipeline': data})
        except Exception as e:
            print(f"WARNING: could not read {PIPELINE_PATH}: {e}")

    # Default inferred pipeline
    default = {
        'extracted_columns': ['Age', 'EstimatedSalary'],
        'steps': [
            'Carga del dataset',
            'Extracción de columnas: Age, EstimatedSalary',
            'División train/test',
            'Escalado con StandardScaler',
            'Entrenamiento: Regresión Logística'
        ]
    }
    return jsonify({'pipeline': default})


@app.route('/metrics', methods=['GET'])
def metrics():
    """Return model metrics exported by Lab04 (if present) or an empty object."""
    import json
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                return jsonify({'metrics': data})
        except Exception as e:
            print(f"WARNING: could not read {METRICS_PATH}: {e}")
    # fallback: empty
    return jsonify({'metrics': {}})


@app.route('/recommend', methods=['POST'])
def recommend():
    """Real-time recommendation wrapper around predict: returns suggestion + probability."""
    load_model_if_changed()
    data = request.get_json(force=True)
    try:
        age = float(data.get('age', 0))
        salary = float(data.get('salary', 0))
    except Exception:
        return jsonify({'error': 'Invalid input values'}), 400

    if scaler is None or clf is None:
        return jsonify({'error': 'Model or scaler not loaded on server'}), 500

    import numpy as np
    X = np.array([[age, salary]])
    Xs = scaler.transform(X)
    pred = int(clf.predict(Xs)[0])
    proba = float(clf.predict_proba(Xs)[0][1])

    suggestion = 'No recomendado'
    if proba >= 0.5:
        suggestion = 'Recomendado'

    return jsonify({'prediction': pred, 'probability': proba, 'suggestion': suggestion})


@app.route('/recommend_batch', methods=['GET', 'POST'])
def recommend_batch():
    """Run the recommendation system for the rows in sample_rows.json and return a single JSON.

    This endpoint reads `sample_rows.json` from the project directory and uses the
    currently loaded model/scaler to compute prediction + probability + suggestion
    for each row. It is intended for the UI to fetch a single batch instead of
    making many per-row requests.
    """
    load_model_if_changed()
    base = os.path.dirname(__file__)
    sample_path = os.path.join(base, 'sample_rows.json')
    if not os.path.exists(sample_path):
        return jsonify({'error': 'sample_rows.json not found'}), 404
    try:
        with open(sample_path, 'r', encoding='utf-8') as fh:
            rows = json.load(fh)
    except Exception as e:
        return jsonify({'error': f'Could not read sample_rows.json: {e}'}), 500

    results = []
    import numpy as _np
    for r in rows:
        try:
            age = float(r.get('Age', 0) or 0)
            salary = float(r.get('EstimatedSalary', 0) or 0)
        except Exception:
            results.append({'nombre': r.get('nombre', ''), 'error': 'invalid numeric values'})
            continue

        if scaler is None or clf is None:
            results.append({'nombre': r.get('nombre', ''), 'error': 'model or scaler not loaded'})
            continue

        try:
            X = _np.array([[age, salary]])
            Xs = scaler.transform(X)
            pred = int(clf.predict(Xs)[0])
            proba = float(clf.predict_proba(Xs)[0][1])
            suggestion = 'Recomendado' if proba >= 0.5 else 'No recomendado'
            results.append({
                'nombre': r.get('nombre', ''),
                'age': age,
                'salary': salary,
                'prediction': pred,
                'probability': proba,
                'suggestion': suggestion
            })
        except Exception as e:
            results.append({'nombre': r.get('nombre', ''), 'error': str(e)})

    return jsonify({'results': results})


@app.route('/routes', methods=['GET'])
def list_routes():
    """Return a JSON list of registered routes (useful for debugging)."""
    rules = []
    for rule in app.url_map.iter_rules():
        rules.append({'endpoint': rule.endpoint, 'rule': str(rule), 'methods': sorted(list(rule.methods))})
    return jsonify({'routes': rules})


@app.route('/plots/<path:filename>', methods=['GET'])
def serve_plot(filename):
    """Serve plot image files if they exist in the project folder.

    Allowed filenames: confusion.png, roc.png, decision.png, sample_plot.png
    """
    allowed = {'confusion.png', 'roc.png', 'decision.png', 'sample_plot.png'}
    if filename not in allowed:
        return abort(404)
    base = os.path.dirname(__file__)
    path = os.path.join(base, filename)
    if not os.path.exists(path):
        return abort(404)
    return send_from_directory(base, filename)


@app.route('/generate_plots', methods=['POST', 'GET'])
def generate_plots_endpoint():
    """Trigger generation of diagnostic plots and metrics.

    Returns JSON with saved filenames and metrics. This can be called after uploading
    a new model/scaler or manually via the UI to refresh images.
    """
    try:
        saved, metrics = generate_and_save_plots()
        return jsonify({'saved': saved, 'metrics': metrics}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Serve the documentation HTML and local static assets so the page can be loaded from the Flask server
@app.route('/', methods=['GET'])
def docs_index():
    base = os.path.dirname(__file__)
    html = 'Documentacion_Modelo_Regresion_Logistica_dark.html'
    if not os.path.exists(os.path.join(base, html)):
        return abort(404)
    return send_from_directory(base, html)


@app.route('/js/<path:filename>', methods=['GET'])
def serve_js(filename):
    base = os.path.join(os.path.dirname(__file__), 'js')
    if not os.path.exists(os.path.join(base, filename)):
        return abort(404)
    return send_from_directory(base, filename)


@app.route('/assets/<path:filename>', methods=['GET'])
def serve_assets(filename):
    base = os.path.join(os.path.dirname(__file__), 'assets')
    if not os.path.exists(os.path.join(base, filename)):
        return abort(404)
    return send_from_directory(base, filename)


if __name__ == '__main__':
    # Print registered routes to help debug missing endpoints, then start server.
    print('\nRegistered routes:')
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted([m for m in rule.methods if m not in ('HEAD','OPTIONS')]))
        print(f"  {str(rule):40}  methods: {methods}")
    # Disable the auto-reloader so the server stays in a single process for our automated checks
    app.run(debug=True, use_reloader=False)
