# Image Anomaly Detection

Reimplementation en PyTorch du papier **Dinomaly: The Less Is More Philosophy in
Multi-Class Unsupervised Anomaly Detection** (CVPR 2025).

Detection d'anomalies industrielles non supervisee par reconstruction de features:
encodeur ViT pre-entraine gele (DINOv2), bottleneck bruite, decodeur a attention
lineaire, reconstruction lache. Un modele unique est entraine sur toutes les
categories a la fois.

- Papier: <https://arxiv.org/abs/2405.14325>
- Code officiel: <https://github.com/guojiajeremy/Dinomaly>
- Methode et details de cette implementation: `docs/method.md`

## Citation

```bibtex
@inproceedings{guo2025dinomaly,
  title={Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection},
  author={Guo, Jia and Lu, Shuai and Zhang, Weihang and Chen, Fang and Li, Huiqi and Liao, Hongen},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20405--20415},
  year={2025}
}
```
