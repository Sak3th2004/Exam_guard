# Contributing to ExamGuard

## Adding a New Detection Engine

1. Create a new file in `backend/engines/` extending `BaseEngine`
2. Implement `_run_analysis()` returning `EngineOutput`
3. Register the engine in `backend/engines/orchestrator.py`
4. Add display name to `backend/services/report_generator.py`
5. Add the engine to the frontend dashboard in `frontend/src/App.jsx`
6. Add the engine to the engines page tab list

### Engine Template

```python
from engines.base_engine import BaseEngine, EngineOutput

class MyEngine(BaseEngine):
    def __init__(self):
        super().__init__(engine_name="my_engine", requires_gpu=False)

    async def _run_analysis(self, answers, answer_key, student_ids, 
                           center_ids, timing_data, question_texts,
                           ground_truth, **kwargs):
        # Your detection logic here
        self.report_progress(50, "Processing...")
        
        return EngineOutput(
            engine_name=self.engine_name,
            flagged_count=len(flagged_ids),
            flagged_student_ids=flagged_ids,
            result_data={...},
            summary={...},
        )
```

## Code Style

- Python: Type hints everywhere, `async/await` for I/O
- JavaScript: Functional React components, hooks
- CSS: BEM naming convention, CSS custom properties

## Testing

```bash
# E2E test (runs full pipeline)
python tests/test_e2e.py

# Frontend dev server
cd frontend && npm run dev
```
