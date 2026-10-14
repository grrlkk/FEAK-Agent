# 참고문헌 확인 기록 — 2026-09-08

> 최종 [서론·관련연구](../paper_docs/FEAK_TC_INTRO_RELATED_2026-09-09.md)의 인용 확인을 위한 보조 기록이다.
> 기존 관련연구 초안의 중복 본문은 정리하고, 확인한 서지와 동명 논문 주의 사항을 보존했다.
> 현재 방법론은 [최종 방법론](../paper_docs/FEAK_TC_METHOD_FINAL.md)을 따른다.

## 1. 인용 목록 — 제목과 발표 정보

공식 출판본이 확인된 논문은 그 발표 정보를 우선했다. arXiv로 표기한 논문은 여기서 확인한
버전을 의미하며, 다른 학회에 출판되지 않았다고 단정하는 표시는 아니다.

1. **Gooding et al. (2025).** Sian Gooding, Lucia Lopez-Rivilla, Edward Grefenstette.
   [Writing as a testbed for open ended agents](https://arxiv.org/abs/2503.19711).
   arXiv:2503.19711. 본문 근거: open-ended writing, 행동·평가·목표 정렬.

2. **Cao and Ng (2025).** Hannan Cao, Hwee Tou Ng.
   [A Constrained Text Revision Agent via Iterative Planning and Searching](https://aclanthology.org/2025.findings-acl.1377/).
   Findings of ACL 2025, pp. 26859–26882. CRAFT/TRIPS 명칭 주의 사항은 §2 참조.

3. **Xiong et al. (2025).** Ruibin Xiong, Yimeng Chen, Dmitrii Khizbullin, Mingchen Zhuge, Jürgen Schmidhuber.
   [Beyond Outlining: Heterogeneous Recursive Planning for Adaptive Long-form Writing with Language Models](https://aclanthology.org/2025.emnlp-main.1254/).
   EMNLP 2025, pp. 24678–24714. 시스템명: WriteHERE.

4. **Wu et al. (2026).** Yuhao Wu, Yushi Bai, Zhiqiang Hu, Juanzi Li, Roy Ka-Wei Lee.
   [SuperWriter: Reflection-Driven Long-Form Generation with Large Language Models](https://aclanthology.org/2026.findings-acl.428/).
   Findings of ACL 2026, pp. 8790–8812. 최초 arXiv 공개는 2025년이다.

5. **Cao et al. (2025).** Hannan Cao, Hai Ye, Hwee Tou Ng.
   [Rationalize and Align: Enhancing Writing Assistance with Rationale via Self-Training for Improved Alignment](https://aclanthology.org/2025.findings-acl.1383/).
   Findings of ACL 2025, pp. 26967–26982.

6. **Fein et al. (2026).** Daniel Fein, Sebastian Russo, Violet Xiang, Kabir Jolly, Rafael Rafailov, Nick Haber.
   [LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing](https://aclanthology.org/2026.eacl-long.362/).
   EACL 2026, pp. 7740–7755. 최초 arXiv 공개는 2025년이다.

7. **Zhao et al. (2026).** Bingchen Zhao et al.
   [APRES: An Agentic Paper Revision and Evaluation System](https://arxiv.org/abs/2603.03142).
   arXiv:2603.03142. 세부 비교는 원문 §3.2와 §4.2에 근거한다.

8. **Lu et al. (2026).** Xinyi Lu, Kexin Phyllis Ju, Mitchell Dudley, Larissa Sano, Xu Wang.
   [AI-Mediated Feedback Improves Student Revisions: A Randomized Trial with FeedbackWriter in a Large Undergraduate Course](https://arxiv.org/abs/2602.16820).
   arXiv:2602.16820, 확인 버전 v2. 조교의 중재와 학생의 수정이 있는 연구다.

9. **Liu and Litman (2026).** Zhexiong Liu, Diane Litman.
   [Intention-Adaptive LLM Fine-Tuning for Text Revision Generation](https://aclanthology.org/2026.findings-eacl.65/).
   Findings of EACL 2026, pp. 1263–1281. 방법명: Intention-Tuning.

10. **Mao et al. (2025).** Song Mao, Lejun Cheng, Pinlong Cai, Guohang Yan, Ding Wang, Botian Shi.
    [DeepWriter: A Fact-Grounded Multimodal Writing Assistant Based On Offline Knowledge Base](https://arxiv.org/abs/2507.14189).
    arXiv:2507.14189, 확인 버전 v2. 기존 초안의 grounding 문맥에 맞춰 선택한 DeepWriter다.

11. **Sahnan et al. (2026).** Dhruv Sahnan et al.
    [Can LLMs Automate Fact-Checking Article Writing?](https://aclanthology.org/2026.tacl-1.23/).
    TACL, volume 14, pp. 489–509. 시스템명: Qraft. 최초 arXiv 공개는 2025년이다.

12. **Chen et al. (2026).** Bingsen Chen, Boyan Li, Ping Nie, Yuyu Zhang, Xi Ye, Chen Zhao.
    [Beyond Single-shot Writing: Deep Research Agents are Unreliable at Multi-turn Report Revision](https://arxiv.org/abs/2601.13217).
    arXiv:2601.13217. 평가 프레임워크: Mr Dre.

13. **Jang et al. (2026).** Chanwoo Jang, Ganghee Go, Jinyong Yun, Seokho Ahn, Myungsun Shin,
    Ho-Hyun Kil, Sungmin Chang, Do-Guk Kim, Young-Duk Seo.
    [From Evaluation to Feedback: A Feature-Based and LLM-Constrained Tool for Korean Writing Assessment](https://doi.org/10.1145/3748522.3780021).
    SAC’26. 서지·내용은 [폴더 내 원고](../../docs/SAC26_AIED_for_arXiv.pdf)를 기준으로 확인했다.

## 2. 논문 식별 주의 사항

### 2.1 CRAFT와 TRIPS

같은 제목의 논문에서 명칭 표기가 일치하지 않는다. 확인한
[저자 공개 PDF](https://michaelcaohn.github.io/assets/pdf/Writing_Agent.pdf)에는 CRAFT/CORD가,
현재 [ACL 초록](https://aclanthology.org/2025.findings-acl.1377/)과
[공식 저장소](https://github.com/nusnlp/CRAFT)에는 TRIPS/ConsTRev가 나타난다.
따라서 두 개의 별도 선행연구로 세지 않고 CRAFT/TRIPS로 병기했다.
논문의 정식 제목과 Cao and Ng (2025)를 서지 식별자로 사용한다.

### 2.2 DeepWriter 동명 논문

기존 초안은 ‘DeepWriter’라고만 적어 저자와 논문을 단정할 수 없었다. 본문은 기존
evidence/grounding 문맥에 맞는 Mao et al. (2025)을 선택했다. 이는 원저자의 의도를 확인한
것이 아니라, 문맥에 근거한 편집상의 선택이다.

별도로 Wang et al. (2026)의
[DeepWriter: A Multi-Agent Collaboration Framework for Information-rich Ultra-long Book Writing](https://ojs.aaai.org/index.php/AAAI/article/view/40648)이
있다. 이 논문은 AAAI 2026, 40(39), pp. 33593–33601에 발표되었으며,
검색된 지식과 계획을 활용하는 책 규모의 생성 연구다. 두 논문의 저자·발표 정보·실험을
섞어 인용하지 않는다. 기존 초안이 이 연구를 가리켰다면 해당 문단과 참고문헌만 교체하면 된다.
