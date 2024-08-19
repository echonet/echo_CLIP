import torch
import numpy as np
from pathlib import Path
import cv2
import re

zero_shot_prompts = {
    "ejection_fraction": [
        "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
        "LV EJECTION FRACTION IS <#>%. ",
    ],
    "pacemaker": [
        "ECHO DENSITY IN RIGHT VENTRICLE SUGGESTIVE OF CATHETER, PACER LEAD, OR ICD LEAD. ",
        "ECHO DENSITY IN RIGHT ATRIUM SUGGESTIVE OF CATHETER, PACER LEAD, OR ICD LEAD. ",
    ],
    "impella": [
        "AN IMPELLA CATHETER IS SEEN AND THE INLET AREA IS 4.0CM FROM THE AORTIC VALVE AND DOES NOT INTERFERE WITH NEIGHBORING STRUCTURES, CONSISTENT WITH CORRECT IMPELLA POSITIONING. THERE IS DENSE TURBULENT COLOR FLOW ABOVE THE AORTIC VALVE, CONSISTENT WITH CORRECT OUTFLOW AREA POSITION ",
        "AN IMPELLA CATHETER IS SEEN ACROSS THE AORTIC VALVE AND IS TOO CLOSE TO OR ENTANGLED IN THE PAPILLARY MUSCLE AND SUBANNULAR STRUCTURES SURROUNDING THE MITRAL VALVE; REPOSITIONING RECOMMENDED. ",
        "AN IMPELLA CATHETER IS SEEN, HOWEVER THE INLET AREA APPEARS TO BE IN THE AORTA OR NEAR THE AORTIC VALVE; REPOSITIONING IS RECOMMENDED. ",
        "AN IMPELLA CATHETER IS SEEN ACROSS THE AORTIC VALVE AND EXTENDS TOO FAR INTO THE LEFT VENTRICLE; REPOSITIONING RECOMMENDED ",
    ],
    "normal_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA SHOWS A NORMAL RESPIRATORY COLLAPSE CONSISTENT WITH NORMAL RIGHT ATRIAL PRESSURE (3MMHG). ",
    ],
    "elevated_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA DEMONSTRATES LESS THAN 50% COLLAPSE CONSISTENT WITH ELEVATED RIGHT ATRIAL PRESSURE (8MMHG). ",
    ],
    "significantly_elevated_right_atrial_pressure": [
        "THE INFERIOR VENA CAVA DEMONSTRATES NO INSPIRATORY COLLAPSE, CONSISTENT WITH SIGNIFICANTLY ELEVATED RIGHT ATRIAL PRESSURE (>15MMHG). ",
    ],
    "pulmonary_artery_pressure": [
        "ESTIMATED PA SYSTOLIC PRESSURE IS <#>MMHG. ",
        "ESTIMATED PA PRESSURE IS <#>MMHG. ",
        "PA PEAK PRESSURE IS <#>MMHG. ",
    ],
    "severe_left_ventricle_dilation": [
        "SEVERE DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "SEVERE DILATED LEFT VENTRICLE BY VOLUME. ",
        "SEVERE DILATED LEFT VENTRICLE. ",
    ],
    "moderate_left_ventricle_dilation": [
        "MODERATE DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "MODERATE DILATED LEFT VENTRICLE BY VOLUME. ",
        "MODERATE DILATED LEFT VENTRICLE. ",
    ],
    "mild_left_ventricle_dilation": [
        "MILD DILATED LEFT VENTRICLE BY LINEAR CAVITY DIMENSION. ",
        "MILD DILATED LEFT VENTRICLE BY VOLUME. ",
        "MILD DILATED LEFT VENTRICLE. ",
    ],
    "severe_right_ventricle_size": ["SEVERE DILATED RIGHT VENTRICLE. "],
    "moderate_right_ventricle_size": ["MODERATE DILATED RIGHT VENTRICLE. "],
    "mild_right_ventricle_size": ["MILD DILATED RIGHT VENTRICLE. "],
    "severe_left_atrium_size": ["SEVERE DILATED LEFT ATRIUM. "],
    "moderate_left_atrium_size": ["MODERATE DILATED LEFT ATRIUM. "],
    "mild_left_atrium_size": ["MILD DILATED LEFT ATRIUM. "],
    "severe_right_atrium_size": ["SEVERE DILATED RIGHT ATRIUM. "],
    "moderate_right_atrium_size": ["MODERATE DILATED RIGHT ATRIUM. "],
    "mild_right_atrium_size": ["MILD DILATED RIGHT ATRIUM. "],
    "tavr": [
        "A BIOPROSTHETIC STENT-VALVE IS PRESENT IN THE AORTIC POSITION. ",
    ],
    "mitraclip": [
        "TWO MITRACLIPS ARE SEEN ON THE ANTERIOR AND POSTERIOR LEAFLETS OF THE MITRAL VALVE. ",
        "TWO MITRACLIPS ARE NOW PRESENT ON THE ANTERIOR AND POSTERIOR MITRAL VALVE LEAFLETS. ",
        "ONE MITRACLIP IS SEEN ON THE ANTERIOR AND POSTERIOR LEAFLETS OF THE MITRAL VALVE. ",
    ],
}


def compute_binary_metric(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
):
    per_frame_similarities = video_embeddings @ prompt_embeddings.T
    # Average along the candidate dimension and frame dimension
    predictions = per_frame_similarities.mean(dim=-1).mean(dim=-1)

    return predictions


def compute_regression_metric(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_values: torch.Tensor,
):
    per_frame_similarities = (
        video_embeddings @ prompt_embeddings.T
    )  # (N x Frames x Candidates)

    # Sort the candidates by their similarity to the video
    ranked_candidate_phrase_indices = torch.argsort(
        per_frame_similarities, dim=-1, descending=True
    )

    # Convert matrix of indices to their corresponding continuous values.
    prompt_values = torch.tensor(
        prompt_values, device=video_embeddings.device
    )  # (N x Frames x Candidates)
    all_frames_ranked_values = prompt_values[ranked_candidate_phrase_indices]

    # Taking the mean along dim=1 collapses the frames dimension
    avg_frame_ranked_values = all_frames_ranked_values.float().mean(
        dim=1
    )  # (N x Candidates)

    # The median of only the top 20% of predicted values is taken
    # as the final predicted value
    twenty_percent = int(avg_frame_ranked_values.shape[1] * 0.2)
    final_prediction = avg_frame_ranked_values[:, :twenty_percent].median(dim=-1)[0]

    return final_prediction


def crop_and_scale(img, res=(640, 480), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)

    return img


def read_avi(p: Path, res=None) -> np.ndarray:
    cap = cv2.VideoCapture(str(p))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    cap.release()
    return np.array(frames)


## TEXT CLEANING UTILS

removables = re.compile(r"\^|CRLF|‡")

in_text_periods = re.compile(r"(?<=\D)\.|\.(?=\D)")
square_brackets = re.compile(r"[\[\]]")
multi_whitespace = re.compile(r"\s+")
multi_period = re.compile(r"\.+")

select_was = re.compile(r"(?<=\b)WAS(?=\b)")
select_were = re.compile(r"(?<=\b)WERE(?=\b)")
select_and_or = re.compile(r"(?<=\b)AND/OR(?=\b)")
select_normally = re.compile(r"NORMALLY")
select_mildly = re.compile(r"MILDLY")
select_moderately = re.compile(r"MODERATELY")
select_severely = re.compile(r"SEVERELY")
select_pa = re.compile(r"PULMONARY ARTERY")
select_icd_codes = re.compile(r"[A-Z](\d+\.\d*\b)")
select_slash_dates = re.compile(r"\d{2}/\d{2}/\d{4}")
select_dot_dates = re.compile(r"\d{2}\.\d{2}\.\d{4}")

space_before_unit = re.compile(r"\s+(MMHG|MM|CM|%)")
space_period = re.compile(r"\s\.")

space_plus_space = re.compile(r"\s\+\s")
verbose_pressure = re.compile(r"\+CVPMMHG")
add_period = [
    r"THE PEAK TRANSAORTIC GRADIENT IS <#>MMHG",
    r"THE MEAN TRANSAORTIC GRADIENT IS <#>MMHG",
    r"LV EJECTION FRACTION IS <#>%",
    r"ESTIMATED PA PRESSURE IS <#>MMHG",
    r"RESTING SEGMENTAL WALL MOTION ANALYSIS",
    r"THE IVC DIAMETER IS <#>MM",
    r"EST RV/RA PRESSURE GRADIENT IS <#>MMHG",
    r"ESTIMATED PEAK RVSP IS <#>MMHG",
    r"HEART FAILURE, UNSPECIFIED",
    r"CHEST PAIN, UNSPECIFIED",
    r"SINUS OF VALSALVA: <#>CM",
    r"THE PEAK TRANSMITRAL GRADIENT IS <#>MMHG",
    r"THE MEAN TRANSMITRAL GRADIENT IS <#>MMHG",
    r"ASCENDING AORTA <#>CM",
    r"ESTIMATED PA SYSTOLIC PRESSURE IS <#>MMHG",
    r"ICD_CODE SHORTNESS BREATH",
    r"ICD_CODE ABNORMAL ELECTROCARDIOGRAM ECG EKG",
    r"SHORTNESS BREATH",
    r"ABNORMAL ELECTROCARDIOGRAM ECG EKG",
    r"THE LEFT ATRIAL APPENDAGE IS NORMAL IN APPEARANCE WITH NO EVIDENCE OF THROMBUS",
]

select_number = r"(?:\d+\.?\d*)"

add_period = [re.escape(a).replace(re.escape("<#>"), select_number) for a in add_period]
add_period = [f"(?:{a})(?!\.)" for a in add_period]
add_period = "|".join(add_period)
add_period = f"({add_period})"
# print(f"{add_period[:50]} ... {add_period[-50:]}")
add_period = re.compile(add_period)


def clean_text(text):
    if len(text) > 1:
        text = text.upper()
        text = text.strip()
        text = text.replace("`", "'")
        text = removables.sub("", text)

        text = in_text_periods.sub(". ", text)
        text = square_brackets.sub("", text)

        text = select_was.sub("IS", text)
        text = select_were.sub("ARE", text)
        text = select_and_or.sub("AND", text)
        text = select_normally.sub("NORMAL", text)
        text = select_mildly.sub("MILD", text)
        text = select_moderately.sub("MODERATE", text)
        text = select_severely.sub("SEVERE", text)
        text = select_pa.sub("PA", text)
        text = select_slash_dates.sub("", text)
        text = select_dot_dates.sub("", text)
        text = select_icd_codes.sub("", text)

        text = space_before_unit.sub(r"\1", text)
        text = space_period.sub(".", text)
        text = multi_whitespace.sub(" ", text)

        text = space_plus_space.sub("+", text)
        text = verbose_pressure.sub("MMHG", text)

        text = text.strip()
        text = text + " "

        text = add_period.sub(r"\1.", text)
        text = multi_period.sub(".", text)

    return text


select_severity = "|".join(
    ["MODERATE/SEVERE", "MILD/MODERATE", "MILD", "MODERATE", "SEVERE", "VERY SEVERE"]
)
select_severity = f"((?<![A-Za-z])(?:{select_severity}))"
select_number = r"(\d+\.?\d*)"

select_variable = "|".join([select_number, select_severity])
# print(select_variable)
select_variable = re.compile(select_variable)


def extract_variables(string, replace_with="<#>"):
    matches = select_variable.findall(string)
    variables = []
    for match in matches:
        for variable in match:
            if variable:
                variables.append(variable)
    variables_replaced = select_variable.sub(replace_with, string)
    return variables, variables_replaced
