from pathlib import Path
from datetime import datetime

from jsonargparse import auto_cli
from srsly import read_jsonl
from loguru import logger
import polars as pl


def test_latency(log_dir: Path | str) -> None:
    """统计日志文件中的各个时间点，并计算latency，并将结果保存到excel文件中"""
    results = []
    log_dir = Path(log_dir)
    for file in log_dir.iterdir():
        if file.suffix == ".jsonl":
            logger.info(f"处理文件: {file}")
            result = handle_file(file)
            results.append(result)
    df = pl.DataFrame(results)
    # 在最后添加一行mean，仅统计asr_latency, llm_latency, tts_latency, total_latency, 其他为空
    df = df.vstack(
        pl.DataFrame(
            {
                "text": ["平均"],
                "speech start": [""],
                "speech end": [""],
                "transcription start": [""],
                "transcription end": [""],
                "asr latency": [df["asr latency"].mean()],
                "llm start": [""],
                "llm first response": [""],
                "llm first sentence": [""],
                "llm end": [""],
                "llm latency": [df["llm latency"].mean()],
                "tts start": [""],
                "tts first response": [""],
                "tts end": [""],
                "tts latency": [df["tts latency"].mean()],
                "total latency": [df["total latency"].mean()],
            }
        )
    )

    # save results to excel file
    df.write_excel(log_dir / "latency.xlsx")


def handle_file(file_path: Path) -> dict[str, str]:
    """统计日志文件中的各个时间点，并计算latency

    说明：
    - asr_latency: transcription_end - speech_end
    - llm_latency: llm_first_sentence - llm_start
    - tts_latency: tts_first_response - tts_start
    - total_latency: tts_latency + llm_latency + asr_latency

    Args:
        file_path (Path): 日志文件路径
    """
    lines = []
    for line in read_jsonl(file_path):
        lines.append(line)
    if len(lines) == 0:
        return {}
    try:
        # speech start end time
        speech_start_line = find_line("speech start", lines)
        speech_start_time = speech_start_line["timestamp"]
        speech_end_line = find_line("speech end", lines)
        speech_end_time = speech_end_line["timestamp"]

        # transcription start end time
        transcription_start_line = find_line("transcription start", lines)
        transcription_start_time = transcription_start_line["timestamp"]
        transcription_end_line = find_line("transcription end", lines)
        transcription_end_time = transcription_end_line["timestamp"]
        text = transcription_end_line.get("text", None)
        if not text:
            text = file_path.stem.split("_")[0]

        # asr latency
        asr_latency = compute_latency(transcription_end_time, speech_end_time)
        logger.info(f"asr latency: {asr_latency}")

        # llm start end time
        llm_start_line = find_line("llm start", lines)
        llm_start_time = llm_start_line.get("timestamp", None)
        llm_first_response_line = find_line("llm first response", lines)
        llm_first_response_time = llm_first_response_line.get("timestamp", None)
        llm_first_sentence_line = find_line("llm first sentence", lines)
        llm_first_sentence_time = llm_first_sentence_line.get("timestamp", None)
        llm_end_line = find_line("llm end", lines)
        llm_end_time = llm_end_line.get("timestamp", None)
        # llm latency
        llm_latency = compute_latency(llm_first_sentence_time, llm_start_time)
        logger.info(f"llm latency: {llm_latency}")

        # tts start end time
        tts_start_line = find_line("tts start", lines)
        tts_start_time = tts_start_line["timestamp"]
        tts_first_response_line = find_line("tts first response", lines)
        tts_first_response_time = tts_first_response_line["timestamp"]
        tts_end_line = find_line("tts end", lines)
        tts_end_time = tts_end_line["timestamp"]
        # tts latency
        tts_latency = compute_latency(tts_first_response_time, tts_start_time)
        logger.info(f"tts latency: {tts_latency}")

        # total latency
        total_latency = float(asr_latency) + float(llm_latency) + float(tts_latency)
        logger.info(f"total latency: {total_latency}")

        # 处理时间戳
        speech_start_time = process_timestamp(speech_start_time)
        speech_end_time = process_timestamp(speech_end_time)
        transcription_start_time = process_timestamp(transcription_start_time)
        transcription_end_time = process_timestamp(transcription_end_time)
        llm_start_time = process_timestamp(llm_start_time)
        llm_first_response_time = process_timestamp(llm_first_response_time)
        llm_first_sentence_time = process_timestamp(llm_first_sentence_time)
        llm_end_time = process_timestamp(llm_end_time)
        tts_start_time = process_timestamp(tts_start_time)
        tts_first_response_time = process_timestamp(tts_first_response_time)
        tts_end_time = process_timestamp(tts_end_time)
        return {
            "text": text,
            "speech start": speech_start_time,
            "speech end": speech_end_time,
            "transcription start": transcription_start_time,
            "transcription end": transcription_end_time,
            "asr latency": asr_latency,
            "llm start": llm_start_time,
            "llm first response": llm_first_response_time,
            "llm first sentence": llm_first_sentence_time,
            "llm end": llm_end_time,
            "llm latency": llm_latency,
            "tts start": tts_start_time,
            "tts first response": tts_first_response_time,
            "tts end": tts_end_time,
            "tts latency": tts_latency,
            "total latency": total_latency,
        }

    except ValueError as e:
        logger.error(f"Error in {file_path}: {e}")
        return


def find_line(key: str, lines: list[dict]) -> dict:
    for line in lines:
        if line["message"] == key:
            return line
    logger.warning(f"Cannot find {key} in {lines}")
    return {}


def compute_latency(end_time: str, start_time) -> str:
    """计算latency
    Args:
        end_time (str): 结束时间, 2025-04-24T14:53:49.102318 格式
        start_time (str): 开始时间, 2025-04-24T14:53:49.102318 格式
    Returns:
        str: latency
    """
    end_time = convert_to_datetime(end_time)
    start_time = convert_to_datetime(start_time)
    latency = (end_time - start_time).total_seconds()
    return latency


def convert_to_datetime(timestamp: str) -> datetime:
    """将时间戳转换为datetime对象
    Args:
        timestamp (str): 时间戳, 2025-04-24T14:53:49.102318 or 2025-04-27T08:43:32.950443+00:00 格式
    Returns:
        datetime: datetime对象
    """
    if "+" in timestamp:
        timestamp = timestamp.split("+")[0]
    return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")


def process_timestamp(timestamp: str) -> str:
    """将时间戳转换为datetime对象
    Args:
        timestamp (str): 时间戳, 2025-04-24T14:53:49.102318 or 2025-04-27T08:43:32.950443+00:00 格式
    Returns:
        datetime: datetime对象
    """
    if "+" in timestamp:
        timestamp = timestamp.split("+")[0]
    return timestamp


if __name__ == "__main__":
    auto_cli(test_latency)
