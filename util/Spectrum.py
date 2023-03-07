import math

def Tanrantula(ef, ep, nf, np):
    a = ef / (ef + nf)
    b = ep / (ep + np)
    return a / (a+b)

def SBI(ep, ef):
    return 1 - (ep / (ep + ef))

def Ochiai(ef, ep, nf):
    return ef / math.sqrt((ef + ep)*(ef + nf))

def Ochiai2(ef, ep, nf, np):
    return ef*np / math.sqrt((ef+ep)*(nf+np)*(ef+np)*(nf+ep))

def Jaccard(ef, ep, nf):
    return ef / (ef + ep + nf)

def Kulczynski(ef, ep, nf):
    return ef / (nf + ep)

def Op2(ef, ep, np):
    return ef - ep/(ep+np+1)

def Dstar2(ef, ep, nf):
    return ef * ef / (ep + nf)


