# Ranker validation on 752 measured MICs (base rate ≤16 µM 64.1%, ≤4 µM 27.4%, median 8.94 µM)

| ranker | Spearman vs log MIC | k | ≤16 µM | ≤4 µM | median µM |
| --- | --- | --- | --- | --- | --- |
| AMPredictor | 0.191 | 25 | 76% | 56% | 4.00 |
| AMPredictor | 0.191 | 50 | 78% | 52% | 4.00 |
| AMPredictor | 0.191 | 100 | 74% | 42% | 5.94 |
| AMPredictor | 0.191 | 200 | 72% | 37% | 7.91 |
| TabPFN | 0.359 | 25 | 92% | 80% | 1.00 |
| TabPFN | 0.359 | 50 | 92% | 76% | 1.48 |
| TabPFN | 0.359 | 100 | 85% | 60% | 3.73 |
| TabPFN | 0.359 | 200 | 80% | 46% | 4.73 |
| ensemble (mean rank) | 0.313 | 25 | 88% | 80% | 2.96 |
| ensemble (mean rank) | 0.313 | 50 | 84% | 62% | 4.00 |
| ensemble (mean rank) | 0.313 | 100 | 85% | 55% | 4.00 |
| ensemble (mean rank) | 0.313 | 200 | 77% | 44% | 5.26 |
