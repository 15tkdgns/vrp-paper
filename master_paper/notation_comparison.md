# 비표준 표기 수정 전후 비교

Word 렌더링 기준: `_{...}` → 아래첨자(작은 글자), `^{...}` → 위첨자(작은 글자)

---

## Case 1. HAR 수식 계수 첨자 (ch2.txt:38)

### 현재 (수정 전)
```
RV_{t+1} = β₀ + β_d · RV_t + β_w · RV̄_{t-4:t} + β_m · RV̄_{t-21:t} + ε_{t+1}
```

**Word 렌더링:**

> RV<sub>t+1</sub> = β<sub>0</sub> + β_d · RV_t + β_w · RV̄<sub>t-4:t</sub> + β_m · RV̄<sub>t-21:t</sub> + ε<sub>t+1</sub>

- `_{t+1}`, `₀`, `_{t-4:t}`, `_{t-21:t}` → 아래첨자 ✅
- `_d`, `_t`, `_w`, `_m` → **literal 언더스코어** ❌ (β_d, RV_t로 출력)

---

### 수정 후
```
RV_{t+1} = β₀ + β_{d} · RV_{t} + β_{w} · RV̄_{t-4:t} + β_{m} · RV̄_{t-21:t} + ε_{t+1}
```

**Word 렌더링:**

> RV<sub>t+1</sub> = β<sub>0</sub> + β<sub>d</sub> · RV<sub>t</sub> + β<sub>w</sub> · RV̄<sub>t-4:t</sub> + β<sub>m</sub> · RV̄<sub>t-21:t</sub> + ε<sub>t+1</sub>

- 모든 첨자 통일 ✅

---

## Case 2. 표 2 — IV_t² (ch3.txt:64)

### 현재 (수정 전)
```
IV_t² | Implied variance proxy, (VIX_t / 100)²
```

**Word 렌더링:**

> IV_t<sup>2</sup> | Implied variance proxy, (VIX_t / 100)<sup>2</sup>

- `²` → 위첨자 ✅
- `_t` → **literal 언더스코어** ❌

---

### 수정 후
```
IV_{t}² | Implied variance proxy, (VIX_{t} / 100)²
```

**Word 렌더링:**

> IV<sub>t</sub><sup>2</sup> | Implied variance proxy, (VIX<sub>t</sub> / 100)<sup>2</sup>

- 아래첨자·위첨자 모두 처리 ✅

---

## Case 3. 본문 내 RV_t, r_t, P_t (ch2, ch3 본문 산재)

### 현재 (수정 전) — 예시: ch2.txt:40 본문
```
여기서 RV_t는 당일 실현분산, RV̄_{t-4:t}는 직전 5거래일 평균, RV̄_{t-21:t}는 직전 22거래일 평균이다.
```

**Word 렌더링:**

> 여기서 RV_t는 당일 실현분산, RV̄<sub>t-4:t</sub>는 직전 5거래일 평균, RV̄<sub>t-21:t</sub>는 직전 22거래일 평균이다.

- `RV_t` → **RV_t (언더스코어 그대로)** ❌
- `_{t-4:t}` → 아래첨자 ✅ (중괄호 있는 것만 처리)

---

### 수정 후 — 예시
```
여기서 RV_{t}는 당일 실현분산, RV̄_{t-4:t}는 직전 5거래일 평균, RV̄_{t-21:t}는 직전 22거래일 평균이다.
```

**Word 렌더링:**

> 여기서 RV<sub>t</sub>는 당일 실현분산, RV̄<sub>t-4:t</sub>는 직전 5거래일 평균, RV̄<sub>t-21:t</sub>는 직전 22거래일 평균이다.

---

## Case 4. RV_i,t(h) 본문 내 (ch3, ch4 본문 산재)

### 현재 (수정 전) — 예시
```
RV_i,t(h) = (252 / h) × Σ_{k=1}^{h} r_{i,t+k}²
```

**Word 렌더링:**

> RV_i,t(h) = (252 / h) × Σ<sub>k=1</sub><sup>h</sup> r<sub>i,t+k</sub><sup>2</sup>

- `RV_i,t(h)` → **literal** ❌ (중괄호 없는 복합 첨자)
- `_{k=1}`, `^{h}`, `_{i,t+k}` → 정상 처리 ✅

---

### 수정 후
```
RV_{i,t}(h) = (252 / h) × Σ_{k=1}^{h} r_{i,t+k}²
```

**Word 렌더링:**

> RV<sub>i,t</sub>(h) = (252 / h) × Σ<sub>k=1</sub><sup>h</sup> r<sub>i,t+k</sub><sup>2</sup>

---

## 수정 범위 요약

| Case | 수정 대상 | 건수 | 난이도 |
|------|----------|------|--------|
| 1. HAR 계수 `β_{d}`, `β_{w}`, `β_{m}`, `RV_{t}` | ch2.txt:38 수식 1줄 + 본문 서술 | ~10건 | 쉬움 |
| 2. 표 2 `IV_{t}²` | ch3.txt:64 표 1줄 | 1건 | 쉬움 |
| 3. 본문 `RV_t`, `r_t`, `P_t` | ch2~ch4 본문 전체 | ~30건 | 오탐 주의 |
| 4. 본문 `RV_i,t(h)` | ch3~ch4 본문 전체 | ~10건 | 쉬움 |

> **주의**: Case 3 본문 `_t` 교체는 일반 단어 안의 언더스코어(예: `r_t`→`r_{t}`)를 자동 치환 시 `adjust_t`, `weight_t` 등 의도치 않은 부분까지 바뀔 수 있어 수동 확인 필요.
