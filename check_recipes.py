import requests

SERVICE_KEY = "a005b424b1be4eaaa1d5"
BASE_URL = f"http://openapi.foodsafetykorea.go.kr/api/{SERVICE_KEY}/COOKRCP01/json"

total = 0
for start in range(1, 1157, 100):
    end = min(start + 99, 1156)
    data = requests.get(f"{BASE_URL}/{start}/{end}").json()['COOKRCP01']
    rows = data.get('row', [])
    total += len(rows)
    print(f"{start:>4}~{end:>4}: {len(rows):>3}개 | total_count={data.get('total_count')} | {data.get('RESULT')}")

print(f"\n합계: {total}개")