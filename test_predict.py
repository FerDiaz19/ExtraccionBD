import json, urllib.request, time, sys

data = json.dumps({'age': 30, 'salary': 40000}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:5000/predict', data=data, headers={'Content-Type':'application/json'})
for i in range(6):
    try:
        resp = urllib.request.urlopen(req, timeout=5).read().decode()
        print(resp)
        sys.exit(0)
    except Exception as e:
        print('attempt', i, 'failed', e)
        time.sleep(1)
print('no response')
sys.exit(1)
