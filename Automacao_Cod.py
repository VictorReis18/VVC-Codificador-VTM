import os
import subprocess
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# =======================
# CONFIGURAÇÃO GERAL
# =======================

NUM_NUCLEOS = 4  # <<< DEFINA AQUI QUANTOS NÚCLEOS QUER USAR

diretorio_base = "/home/victor/VVCSoftware_VTM-VTM-23.0/VVCSoftware_VTM-VTM-23.0"
config_randomaccess = "cfg/encoder_randomaccess_vtm.cfg"
config_videos_base = os.path.join(diretorio_base, "cfg/per-sequence")
videos = ["BasketballPass", "BlowingBubbles", "BQSquare"]
qps = [22, 27, 32, 37]


# =======================
# FUNÇÕES AUXILIARES
# =======================

def obter_frames_do_cfg(caminho_cfg):
    try:
        with open(caminho_cfg, "r") as arquivo:
            for linha in arquivo:
                match = re.search(r"FramesToBeEncoded\s*:\s*(\d+)", linha)
                if match:
                    return int(match.group(1))
    except FileNotFoundError:
        print(f"Arquivo de configuração não encontrado: {caminho_cfg}")
    return None


def executar_coding(args):
    """Função executada em paralelo."""
    nome_video, qp, frames, caminho_cfg, diretorio_video = args

    diretorio_logs = os.path.join(diretorio_video, "logs")
    diretorio_yuvs = os.path.join(diretorio_video, "yuvs")
    os.makedirs(diretorio_logs, exist_ok=True)
    os.makedirs(diretorio_yuvs, exist_ok=True)

    caminho_saida_yuv = os.path.join(diretorio_yuvs, f"{nome_video}_qp{qp}.yuv")
    caminho_saida_log = os.path.join(diretorio_logs, f"{nome_video}_qp{qp}.log")

    comando = (
        f"./bin/EncoderAppStatic "
        f"-c {config_randomaccess} "
        f"-c {caminho_cfg} "
        f"-f {frames} -q {qp} "
        f"-o {caminho_saida_yuv} > {caminho_saida_log}"
    )

    try:
        subprocess.run(
            comando,
            shell=True,
            cwd=diretorio_base,  # <<< Executa no diretório correto
            check=True
        )
        return f"[OK] {nome_video} | QP={qp} | Frames={frames}"
    except subprocess.CalledProcessError as e:
        return f"[ERRO] {nome_video} | QP={qp} | Erro: {e}"


# =======================
# SCRIPT PRINCIPAL
# =======================

def codificar_videos():

    diretorio_resultados = os.path.join(diretorio_base, "resultados")
    os.makedirs(diretorio_resultados, exist_ok=True)

    print("Escolha o método de definição da quantidade de frames:")
    print("1 - Utilizar a quantidade de frames do arquivo .cfg")
    print("2 - Definir quantidade personalizada de frames")
    escolha = input("Digite 1 ou 2: ").strip()

    if escolha == "2":
        frames_personalizados = input("Digite a quantidade de frames: ").strip()
        if not frames_personalizados.isdigit():
            print("Valor inválido. Encerrando.")
            return
        frames_personalizados = int(frames_personalizados)
    elif escolha != "1":
        print("Escolha inválida. Encerrando.")
        return

    tarefas = []

    # Preparar todas as tarefas antes da execução
    for nome_video in videos:

        caminho_cfg = os.path.join(config_videos_base, f"{nome_video}.cfg")
        frames_cfg = obter_frames_do_cfg(caminho_cfg)

        if frames_cfg is None:
            print(f"Skip: não foi possível ler o CFG de {nome_video}.")
            continue

        frames = frames_personalizados if escolha == "2" else frames_cfg
        diretorio_video = os.path.join(diretorio_resultados, nome_video)
        os.makedirs(diretorio_video, exist_ok=True)

        for qp in qps:
            tarefas.append((nome_video, qp, frames, caminho_cfg, diretorio_video))

    print(f"\nIniciando paralelização com {NUM_NUCLEOS} núcleos...")
    print(f"Total de tarefas: {len(tarefas)}\n")

    # EXECUÇÃO PARALELA
    with ProcessPoolExecutor(max_workers=NUM_NUCLEOS) as executor:
        futuros = {executor.submit(executar_coding, t): t for t in tarefas}

        for futuro in as_completed(futuros):
            print(futuro.result())

    print("\nProcessamento concluído!")


if __name__ == "__main__":
    codificar_videos()
