import numpy as np

# Tabela para converter numeros atomicos em simbolos e raios covalentes (Angstroms)
NUM_PARA_SIMBOLO = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    24: "Cr",
    35: "Br",
    53: "I",
}

RAIOS_COVALENTES = {
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Cr": 1.39,
    "Br": 1.20,
    "I": 1.39,
}


def ler_xyz_sem_hidrogenios(caminho_arquivo):
    """Le o arquivo .xyz (com ou sem cabecalho) e descarta TODOS os atomos de hidrogenio."""
    atomos_pesados = []
    coordenadas_pesadas = []

    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        linhas = [linha.strip() for linha in f.readlines() if linha.strip()]

    # Detecta se ha cabecalho padrao XYZ ou se as coordenadas comecam na primeira linha
    if len(linhas[0].split()) == 1 and linhas[0].isdigit():
        num_atomos = int(linhas[0])
        linhas_dados = linhas[2 : 2 + num_atomos]
    else:
        linhas_dados = linhas

    for linha in linhas_dados:
        partes = linha.split()
        if len(partes) >= 4:
            ident = partes[0]
            # Converte numero atomico (ex: '1' ou '24') para simbolo ('H' ou 'Cr')
            if ident.isdigit():
                simbolo = NUM_PARA_SIMBOLO.get(int(ident), ident)
            else:
                simbolo = ident.capitalize()

            # Descarta os hidrogenios
            if simbolo != "H":
                atomos_pesados.append(simbolo)
                coordenadas_pesadas.append(
                    [float(partes[1]), float(partes[2]), float(partes[3])]
                )

    return atomos_pesados, np.array(coordenadas_pesadas)


def kabsch_alinhamento(P_calc, Q_exp):
    """Calcula o RMSD minimo (Eq. 3) e a translacao t sobre os atomos pesados."""
    if P_calc.shape != Q_exp.shape:
        raise ValueError(
            f"Numero de atomos pesados diferente: P={P_calc.shape[0]}, Q={Q_exp.shape[0]}"
        )

    # Centroides dos atomos pesados
    p_barra = np.mean(P_calc, axis=0)
    q_barra = np.mean(Q_exp, axis=0)

    # Pre-centralizacao
    P_centrado = P_calc - p_barra
    Q_centrado = Q_exp - q_barra

    # Covariancia e SVD
    H = P_centrado.T @ Q_centrado
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Rotacao otima R
    d = np.sign(np.linalg.det(V @ U.T))
    R = V @ np.diag([1.0, 1.0, d]) @ U.T

    # Vetor de translacao t e sua magnitude ||t||
    t = q_barra - (R @ p_barra)
    grau_translacao = np.linalg.norm(t)

    # Alinhamento completo e calculo do RMSD (Eq. 3)
    P_alinhado = (P_calc @ R.T) + t
    rmsd = np.sqrt(np.mean(np.sum((P_alinhado - Q_exp) ** 2, axis=1)))

    return rmsd, t, grau_translacao


def calcular_mud_msd(atomos, coords_calc, coords_exp, tolerancia=0.45):
    """Calcula MUD (Eq. 4) e MSD (Eq. 5) para as M ligacoes entre atomos pesados."""
    diferencas = []
    n = len(atomos)

    for i in range(n):
        for j in range(i + 1, n):
            r1 = RAIOS_COVALENTES.get(atomos[i], 1.20)
            r2 = RAIOS_COVALENTES.get(atomos[j], 1.20)
            dist_exp = np.linalg.norm(coords_exp[i] - coords_exp[j])

            if dist_exp <= (r1 + r2 + tolerancia):
                dist_calc = np.linalg.norm(coords_calc[i] - coords_calc[j])
                diferencas.append(dist_calc - dist_exp)

    diferencas = np.array(diferencas)
    return len(diferencas), np.mean(np.abs(diferencas)), np.mean(diferencas)


# --- Execucao ---
if __name__ == "__main__":
    # Estrutura experimental de Raio-X (referencia)
    atomos_exp, Q_exp = ler_xyz_sem_hidrogenios("2282422_tratada.xyz")

    # Estrutura otimizada no Gaussian/ORCA
    atomos_calc, P_calc = ler_xyz_sem_hidrogenios("2282422_wb97xd3_tzvp.xyz")

    if atomos_exp != atomos_calc:
        print("Aviso: A ordem dos atomos pesados difere entre os arquivos!")

    rmsd, t, grau_t = kabsch_alinhamento(P_calc, Q_exp)
    num_lig, mud, msd = calcular_mud_msd(atomos_exp, P_calc, Q_exp)

    print(f"Atomos pesados considerados (N): {len(Q_exp)}")
    print(f"Ligacoes entre atomos pesados (M): {num_lig}")
    print("-" * 48)
    print(f"RMSD (Eq. 3): {rmsd:.4f} Angstroms")
    print(f"MUD  (Eq. 4): {mud:.4f} Angstroms")
    print(f"MSD  (Eq. 5): {msd:+.4f} Angstroms")
    print(f"Grau de translacao (||t||): {grau_t:.4f} Angstroms")
