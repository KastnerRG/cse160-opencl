from abc import ABC, abstractmethod
from pathlib import Path
from subprocess import Popen, PIPE
from typing import List, Dict, Any
from prof_base import ProfBase
import json

class InterceptLayerProfBase(ProfBase, ABC):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self) -> float:
        json_object = self.profile()

        # Look for Completed events
        completed_events = [event for event in json_object if event.get("ph") == "X"]
        student_kernel_events = [event for event in completed_events if not event.get("name").startswith("cl")]

        # Duration is in microseconds, convert to milliseconds
        return sum(event.get("dur", 0.0) * 1.0E-3 for event in student_kernel_events)
    
    @abstractmethod
    def profile(self) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
class InterceptLayerProfExecutable(InterceptLayerProfBase):
    def __init__(self, args: List[str]) -> None:
        super().__init__()

        self._args = args

    def profile(self) -> List[Dict[str, Any]]:
        process = Popen(["/bin/cliloader", "--dump-dir", ".", "-ckt"] + self._args, stderr=PIPE, text=True)
        process.wait()

        clintercept_trace_path = Path("clintercept_trace.json")
        clintercept_report_path = Path("clintercept_report.txt")

        with clintercept_trace_path.open("r") as f:
            json_object = json.loads(f.read())

        clintercept_trace_path.unlink()
        clintercept_report_path.unlink()

        return json_object
    
class InterceptLayerProfFile(InterceptLayerProfBase):
    def __init__(self, path: Path) -> None:
        super().__init__()

        self._path = path

    def profile(self) -> List[Dict[str, Any]]:
        with self._path.open("r") as f:
            return json.loads(f.read())

class InterceptLayerProfTest(InterceptLayerProfBase):
    def __init__(self) -> None:
        super().__init__()

    def profile(self) -> List[Dict[str, Any]]:
        json_str = """
[
{"ph":"M","name":"process_name","pid":825,"tid":0,"args":{"name":"solution"}},
{"ph":"M","name":"clintercept_start_time","pid":825,"tid":0,"args":{"start_time":7334998904}},
{"ph":"M","name":"thread_name","pid":825,"tid":1.1,"args":{"name":"IOQ 0xaaab0630b770.0.0 cpu-cortex-a53-0x000 (CL_DEVICE_TYPE_CPU)"}},
{"ph":"M","name":"thread_sort_index","pid":825,"tid":1.1,"args":{"sort_index":"1"}},
{"ph":"X","pid":825,"tid":"clEnqueueWriteBuffer","name":"clEnqueueWriteBuffer","ts":923286.187,"dur":11918.584,"args":{"id":0}},
{"ph":"X","pid":825,"tid":"clEnqueueWriteBuffer","name":"clEnqueueWriteBuffer","ts":936294.292,"dur":8959.667,"args":{"id":1}},
{"ph":"X","pid":825,"tid":"matrixMultiply","name":"matrixMultiply","ts":946505.396,"dur":2569820.334,"args":{"id":2}},
{"ph":"X","pid":825,"tid":"clEnqueueReadBuffer","name":"clEnqueueReadBuffer","ts":3516359.168,"dur":9089.500,"args":{"id":3}},
{"ph":"M","name":"clintercept_eof","pid":825,"tid":0}
]
"""

        return json.loads(json_str)
        

if __name__ == "__main__":
    test = InterceptLayerProfTest()
    print(test())
