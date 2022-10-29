from sentence_transformers import SentenceTransformer, util
from website.nlp_kng import config
import sentence_transformers
import torch

@config.timer
def gen_qa():
    text = ["two sources of information (parent reported sleep diaries and actigraph recordswere used to investigate how toddler sleep characteristics (bed time/sleep onset, wake time/sleep offset, total nighttime sleep and total sleep timeare related to sleep problems and temperament.",
    "two sources of information (parent reported sleep diaries and actigraph recordswere used to investigate how toddler sleep characteristics (bed time/sleep onset, wake time/sleep offset, total nighttime sleep and total sleep timeare related to sleep problems and temperament.",
    " the findings that parent reported and actigraph recorded sleep characteristics varied as a function of parent report of toddler sleep problems and temperament add needed information on toddler sleep.",
    " bates, viken, alexander, beyers, and stockton (2002found that 4- to 5-year-old children with disrupted sleep, based on parent daily sleep diaries documenting the variability in sleep duration and bedtimes, had more teacher-reported adjustment problems in preschool.",
    "for both groups, actigraph records did not verify parent reports of sleep problems. however, the authors speculated that children’s behaviors related to sleep, such as bedtime struggles and variability in sleep patterns, contributed to, and may have exaggerated, parental perceptions of their children’s sleep problems.",
    "however, the authors speculated that children’s behaviors related to sleep, such as bedtime struggles and variability in sleep patterns, contributed to, and may have exaggerated, parental perceptions of their children’s sleep problems.",
    " the national sleep foundation (2014estimates that toddlers 1- to 3-years-old should get 12 to 14 hours of sleep in a 24-hour period. parents use sleep diaries to record daily information across a specified period of time (typically from a few days to several weeksabout a child’s bedtimes, morning wake times, and sleep disruptions (e.g., when the child was 'out of bed')."]
    model = sentence_transformers.models.Transformer("sentence-transformers/"+config.QA_BASE_MPNET_MODEL, model_args={"use_fast": False})
    output = []
    for sent in text:
        print(f'gen_qa_mpnet_embeddings : {sent}')
        output_tensor = model.encode(sent, show_progress_bar = False, convert_to_tensor = True)
        output.append(output_tensor)
    result = torch.Tensor(torch.stack(output, dim=1))
    # return result
    return

gen_qa()
