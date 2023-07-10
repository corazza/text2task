import random
import re
from collections import Counter
from typing import Any, Tuple

import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from nltk.metrics.distance import edit_distance
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from transformers import pipeline

import compiler_interface
from consts import *
from datasets_common import load_terms
from reward_machine import RewardMachine


def get_generator(model_path: str):
    return pipeline('text-generation', model=model_path)


def answer_query_single(model_path: str, do_cluster: bool, displayer) -> tuple[RewardMachine, str, str]:
    generator = get_generator(model_path)
    return answer_query(generator, do_cluster, displayer)


def answer_query(generator, do_cluster: bool, displayer) -> tuple[RewardMachine, str, str]:
    desc: str = input(': ')
    rm: RewardMachine
    src: str
    rm, src = get_rm(generator, desc, do_cluster, displayer)
    return rm, desc, src


def query_loop(model_path: str, do_cluster: bool, displayer) -> tuple[RewardMachine, str, str]:
    generator = get_generator(model_path)
    print('Please input the task')
    reward_machine: RewardMachine
    src: str
    reward_machine, desc, src = answer_query(generator, do_cluster, displayer)
    print(f'{src}')
    return reward_machine, desc, src


def similarity_score_output(desc: str, output: str) -> float:
    desc_words: set[str] = set(re.findall(r'\b\w+\b', desc))
    output_words: set[str] = set(re.findall(
        r'\b\w+\b', output)) - set(['some'])
    diff: set[str] = desc_words ^ output_words
    return -len(diff)


def compile_filter(outputs: list[str]) -> list[tuple[RewardMachine, str]]:
    results: list[tuple[RewardMachine, str]] = []
    for src in outputs:
        try:
            rm: RewardMachine = compiler_interface.compile(src)
            results.append((rm, src))
        except Exception as e:
            continue
    return results


def levenshtein_distance(out1: tuple[RewardMachine, str], out2: tuple[RewardMachine, str]) -> float:
    return edit_distance(out1[1], out2[1])


def generate_trace(appears: frozenset[str], length: int) -> list[frozenset[str]]:
    result: list[frozenset[str]] = [frozenset()]
    appears_list: list[str] = list(appears)
    chosen_length = max(2, length)
    for i in range(chosen_length):
        to_add: set[str] = set()
        repeats: int = list(range(len(appears)))
        num_repeat: int = 1 + np.random.choice([0] * len(repeats) + repeats)
        for j in range(num_repeat):
            to_add.add(np.random.choice(appears_list))
        result.append(frozenset(to_add))
        result.append(frozenset())
    return result


def score_trace(rms: list[RewardMachine], trace: list[frozenset[str]]) -> float:
    num_pos: int = 0
    for rm in rms:
        if rm.reward_sum(trace) > 0:
            num_pos += 1
    return abs(0.5 - float(num_pos)/float(len(rms)))


def get_traces(rms: list[RewardMachine], displayer) -> list[list[frozenset[str]]]:
    displayer('[SEMANTIC CLUSTERING]\ngetting traces...')
    appears: frozenset[str] = get_appears(rms)
    traces: list[list[frozenset[str]]] = []
    num_states: list[int] = [len(x.get_nonterminal_states()) for x in rms]
    for i in range(SEMANTIC_SIMILARITY_SAMPLES_REDUNDANCY*SEMANTIC_SIMILARITY_NUM_SAMPLES):
        # num_local_appears: int = 1 + np.random.choice(range(len(appears)))
        # local_appears: frozenset[str] = frozenset(
        #     random.sample(appears, num_local_appears))
        trace: list[frozenset[str]] = generate_trace(
            appears, 1 + np.random.choice(range(np.random.choice(num_states))))
        traces.append(trace)
    displayer('[SEMANTIC CLUSTERING]\nscoring traces...')
    scored_traces: list[tuple[list[frozenset[str]], float]] = [
        (trace, score_trace(rms, trace)) for trace in traces]
    sorted_traces: list[tuple[list[frozenset[str]], float]] = sorted(
        scored_traces, key=lambda x: x[1])
    top_traces: list[tuple[list[frozenset[str]], float]
                     ] = sorted_traces[:SEMANTIC_SIMILARITY_NUM_SAMPLES]
    result: list[list[frozenset[str]]] = [x[0] for x in top_traces]
    result.append([frozenset()] * 10)
    return result


def semantic_distance(traces: list[list[frozenset[str]]], out1: tuple[RewardMachine, str], out2: tuple[RewardMachine, str]) -> float:
    rm1: RewardMachine = out1[0]
    rm2: RewardMachine = out2[0]
    num_same: int = 0
    for trace in traces:
        r1 = rm1.reward_sum(trace)
        r2 = rm2.reward_sum(trace)
        if (r1 > 0 and r2 > 0) or (r1 == 0 and r2 == 0):
            num_same += 1
    return 1 - float(num_same) / float(len(traces))


def get_appears(outputs: list[RewardMachine]) -> frozenset[str]:
    result: set[str] = set()
    for rm in outputs:
        result.update(rm.appears)
    return frozenset(result)


def display_cluter(dist_matrix):
    dist_array = squareform(dist_matrix)
    linkage_matrix = sch.linkage(dist_array, method='average')

    plt.figure()

    dendrogram = sch.dendrogram(linkage_matrix)

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data point')
    plt.ylabel('Distance')

    plt.show()


def cluster(outputs: list[tuple[RewardMachine, str]],
            distance_f,
            displayer) -> tuple[tuple[RewardMachine, str], Any, Any]:
    if len(outputs) == 1:
        return outputs[0]
    rms: list[RewardMachine] = [o[0] for o in outputs]
    traces = get_traces(rms, displayer)
    dist_matrix = np.zeros((len(outputs), len(outputs)))
    displayer('[SEMANTIC CLUSTERING]\ncomputing distance matrix...')
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            dist_matrix[i, j] = distance_f(traces, outputs[i], outputs[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    displayer('[SEMANTIC CLUSTERING]\nclustering...')
    num_clusters: int = min(SEMANTIC_SIMILARITY_NUM_CLUSTERS, len(outputs))
    cluster_model = AgglomerativeClustering(
        n_clusters=num_clusters, affinity='precomputed', linkage='average')
    clusters = cluster_model.fit_predict(dist_matrix)
    # most_frequent = np.bincount(clusters).argmax()
    counter = Counter(clusters)
    most_frequent = max(clusters, key=lambda x: (
        counter[x], -np.where(clusters == x)[0][0]))
    for output, cluster in zip(outputs, clusters):
        if cluster == most_frequent:
            return output, dist_matrix, traces
    assert False, "typing, this shouldn't ever happen"


def synthesize(generator, desc: str, do_cluster: bool, displayer) -> str:
    prompt = f'<|bos|>{desc}<|sep|>'
    displayer('[MODEL]\ncalling model...')
    model_outputs = generator(prompt,
                              max_new_tokens=300,
                              bos_token_id=generator.tokenizer.sep_token_id,
                              eos_token_id=generator.tokenizer.eos_token_id,
                              pad_token_id=generator.tokenizer.pad_token_id,
                              num_return_sequences=MODEL_NUM_RETURN_SEQUENCES,
                              do_sample=False,
                              num_beams=MODEL_NUM_RETURN_SEQUENCES,
                              temperature=MODEL_TEST_TEMPERATURE,
                              #   num_beam_groups=2,
                              return_full_text=False,
                              )
    outputs: list[str] = [output['generated_text'] for output in model_outputs]
    # for o in outputs:
    #     print(o)
    outputs_rm: list[tuple[RewardMachine, str]] = compile_filter(outputs)
    scored_outputs: list[tuple[RewardMachine, str, float]] = [
        (output[0], output[1], similarity_score_output(desc, output[1])) for output in outputs_rm]
    sorted_outputs: list[tuple[RewardMachine, str, float]] = sorted(
        scored_outputs, key=lambda x: -x[2])
    try:
        top_score: float = sorted_outputs[0][2]
    except:
        print('failed')
        IPython.embed()
    # top_brass: list[tuple[RewardMachine, str]] = [
    #     (x[0], x[1]) for x in sorted_outputs if x[2] == top_score]
    top_brass_with_score: list[tuple[RewardMachine, str, float]
                               ] = sorted_outputs[:int(MODEL_NUM_RETURN_SEQUENCES/2)]
    top_brass: list[tuple[RewardMachine, str]] = [
        (x[0], x[1]) for x in top_brass_with_score]
    # IPython.embed()
    if do_cluster:
        result, dist_matrix, traces = cluster(
            top_brass, semantic_distance, displayer)
        IPython.embed()
        # display_cluter(dist_matrix)
        return result[1]
    else:
        return top_brass[0][1]


def get_rm(generator, desc: str, do_cluster: bool, displayer) -> Tuple[RewardMachine, str]:
    for i in range(10):
        src = synthesize(generator, desc, do_cluster, displayer)
        print(desc)
        print(src)
        displayer(f'[MODEL]\n{src}')
        ast = compiler_interface.parse(src)
        nfa, _ = compiler_interface.get_nfa(src)
        dfa, _ = compiler_interface.get_dfa(src)
        return compiler_interface.compile(src), src
    raise Exception(
        f"couldn't produce syntactically valid output in 10 tries, giving up")
