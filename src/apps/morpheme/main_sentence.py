import os, sys, time
import logging
import torch
import spacy
import gc
import parser
import argparse
from torch import nn, optim
from config import *
from models.build_model import build_model
from data import Multi30k
from utils import get_bleu_score, greedy_decode


DATASET = Multi30k()


def train(model, data_loader, optimizer, criterion, epoch, checkpoint_dir):
    model.train()
    epoch_loss = 0

    for idx, (src, tgt) in enumerate(data_loader):
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        tgt_x = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        optimizer.zero_grad()

        output, _ = model(src, tgt_x)

        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = tgt_y.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
    num_samples = idx + 1

    if checkpoint_dir and epoch % 100 == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{epoch:04d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            checkpoint_file,
        )

    return epoch_loss / num_samples


def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0

    total_bleu = []
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(data_loader):
            src = src.to(model.device)
            tgt = tgt.to(model.device)
            tgt_x = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            output, _ = model(src, tgt_x)

            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = tgt_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
            score = get_bleu_score(output, tgt_y, DATASET.vocab_tgt, DATASET.specials)
            total_bleu.append(score)
        num_samples = idx + 1

    loss_avr = epoch_loss / num_samples
    bleu_score = sum(total_bleu) / len(total_bleu)
    return loss_avr, bleu_score


def main(args):
    # 처음부터
    # model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=DEVICE, dr_rate=DROPOUT_RATE)
    # 불러오기
    model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=DEVICE, dr_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load(args.resume_from)["model_state_dict"])
    """
    def initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform_(model.weight.data)
    """
    # model.apply(initialize_weights)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, verbose=True, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.pad_idx)

    train_iter, valid_iter, test_iter = DATASET.get_iter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    for epoch in range(1501, N_EPOCH):
        logging.info(f"*****epoch: {epoch:02}*****")
        train_loss = train(model, train_iter, optimizer, criterion, epoch, CHECKPOINT_DIR)
        logging.info(f"train_loss: {train_loss:.5f}")
        valid_loss, precision = evaluate(model, valid_iter, criterion)
        if epoch > WARM_UP_STEP:
            scheduler.step(valid_loss)
        logging.info(f"valid_loss: {valid_loss:.5f}, precision: {precision:.5f}")

        logging.info(
            DATASET.translate(
                model,
                "2012 년 세계 자연 보전 총회 ( World Conservation Congress ) 유치 에 성공 한 제주도 가 세계 환경 수도 조성 에 잰 걸음 을 내딛 었 다 .",
                greedy_decode,
            )
        )
        # 2012/SN 년/NNB 세계/NNG 자연/NNG 보전/NNG 총회/NNG (/SS World/SL Conservation/SL Congress/SL )/SS 유치/NNG 에/JKB 성공/NNG 하/XSV ㄴ/ETM 제주도/NNP 가/JKS 세계/NNG 환경/NNG 수도/NNG 조성/NNG 에/JKB 잰걸음/NNG 을/JKO 내딛/VV 었/EP 다/EF ./SF

        logging.info(" ")

        # logging.info(DATASET.translate(model, "한 해설 위원 은 “ 일반 적 으로 판타지 스타 ( Fantasista ) 라 하 면 드리블 , 슛 , 패스 능력 을 두루 갖춘 선수 다 .", greedy_decode))
        # 한/MMN 해설/NNG 위원/NNG 은/JX “/SS 일반/NNG 적/XSN 으로/JKB 판타지스타/NNG (/SS Fantasista/SL )/SS 이/VCP 라/EC 하/VV 면/EC 드리블/NNG ,/SP 슛/NNG ,/SP 패스/NNG 능력/NNG 을/JKO 두루/MAG 갖추/VV ㄴ/ETM 선수/NNG 이/VCP 다/EF ./SF

        logging.info(" ")

        logging.info(DATASET.translate(model, "수입 물가 넉 달 만 에 급등", greedy_decode))
        # 수입/NNG 물가/NNG 넉/MMN 달/NNB 만/NNB 에/JKB 급등/NNG

    test_loss, bleu_score = evaluate(model, test_iter, criterion)
    logging.info(f"test_loss: {test_loss:.5f}, bleu_score: {bleu_score:.5f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", default="./chk/1500.pt")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)
