import numpy as np
import scipy as sp
import scipy.linalg
import math

class State(object):
    ###ONE QUBIT STATES###
    #z-basis states
    #|0>=(1,0)
    zero=np.array([[1],[0]])
    #|1>=(0,1)
    one=np.array([[0],[1]])
    
    normalizeState=staticmethod(lambda state: state/sp.linalg.norm(state))
    
    #x-basis states
    #|+>=1/sqrt(2)(|0>+|1>)
    plus=normalizeState.__func__(zero+one)
    #|->=1/sqrt(2)(|0>-|1>)
    minus=normalizeState.__func__(zero-one)

    #y-basis states
    #|R>=norm(|0>+i|1>)
    R=normalizeState.__func__(zero+1j*one)
    #|L>=norm(|0>-i|1>)
    L=normalizeState.__func__(zero-1j*one)
    
    @staticmethod
    def nKron(*args):
        result=np.array([[1]])
        for q in args:
            result=np.kron(result,q)
        return result
    
    #Bell state
    cat=normalizeState.__func__(nKron.__func__(zero,zero)+nKron.__func__(one,one))

    totalQubitsInState=staticmethod(lambda state: math.log2(state.shape[0])/math.log2(2))


class GateFunc(object):
    #Universal Controlled Gate Matrix
    P0=np.dot(State.zero,State.zero.T)
    P1=np.dot(State.one,State.one.T)
    ID=np.eye(2)

    @staticmethod
    def createGate(gate,target,total):
        if target==0:
            result=gate
        else:
            result=GateFunc.ID
        for n in range(1,total):
            if n==target:
                result=State.nKron(result,gate)
            else:
                result=State.nKron(result,GateFunc.ID)
        return result
            
    
    @staticmethod
    def createControlledGate(gate,control,target,total):
        #TODO: data validation checks
        if gate.shape!=(2,2):
            raise ValueError('U must be a 2x2 matrix')
        if control<target:
            result=GateFunc.createControlledGateINT(gate,control,target,total)
        if control>target:
            result=GateFunc.createControlledGateINT(GateFunc.P1,target,control,total,GateFunc.ID,gate,GateFunc.P0)
        return result
    
    @classmethod
    def createControlledGateINT(self,gate,control,target,total,P0=P0,P1=P1,ID=ID):
        C0=GateFunc.createGate(P0,control,control+1)
        C1=GateFunc.createGate(P1,control,control+1)
        X0=GateFunc.createGate(ID,target-control-1,total-control-1)
        X1=GateFunc.createGate(gate,target-control-1,total-control-1)
        result=State.nKron(C0,X0)+State.nKron(C1,X1)
        return result

class Gate(object):
    ###One QUBIT GATE MATRICES###
    #Identity
    ID=np.eye(2)
    #Gloabl Phase (takes parameter "shift")
    @staticmethod
    def gPhase(shift):
        result=np.dot(np.exp(1j*shift),Gate.ID)
        return result
    #Pauli Gates
    pauliX=np.array([[0,1],[1,0]])
    NOT=pauliX
    pauliY=np.array([[0,-1j],[1j,0]])
    pauliZ=np.array([[1,0],[0,1]])
    #Hadamard Gate
    hadamard=(1/np.sqrt(2))*np.array([[1,1],[1,-1]])
    #Phase Gates
    phaseZ=np.array([[1,0],[0,-1]])
    phaseS=sp.linalg.sqrtm(phaseZ)
    phaseT=sp.linalg.sqrtm(phaseS)
    #Rotation Gates (take parameter "shift"):
    Rx=staticmethod(lambda shift: sp.linalg.exp(-1j*pauliX*shift/2))
    Ry=staticmethod(lambda shift: sp.linalg.exp(-1j*pauliY*shift/2))
    Rz=staticmethod(lambda shift: sp.linalg.exp(-1j*pauliZ*shift/2))

    #TODO: Programatically create CCNOT gates and the like
    
    #Common Controlled Gates upt to 3 qubits
    CNOT=dict()
    for total in range(2,4):
        for control in range(3):
            if control<total:
                for target in range(3):
                    if target<total and control!=target:
                        CNOT[control,target,total]=GateFunc.createControlledGate(NOT,control,target,total)
    
    cPhaseS=dict()
    for total in range(2,4):
        for control in range(3):
            if control<total:
                for target in range(3):
                    if target<total and control!=target:
                        cPhaseS[control,target,total]=GateFunc.createControlledGate(phaseS,control,target,total)
    cPhase=dict()
    @staticmethod
    def cPhase(shift):
        for total in range(2,4):
            for control in range(3):
                if control<total:
                    for target in range(3):
                        if target<total and control!=target:
                            cPhase[control,target,total,shift]=GateFunc.createControlledGate(Gate.gPhase(shift),control,target,total)
        return cPhase

    #Swap gate
    #TODO: make this programatically
    SWAP=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])


class Compute(object):
    #TODO: actually make this a computer
    @staticmethod
    def runGate(gate,state):
        if gate.shape[0]==state.shape[0]:
            newState=np.dot(gate,state)
        else:
            raise ValueError('gate size does not match total qubits in state')
        


class Measure(object):
    @staticmethod
    def measure(state,qubit):
        #TODO: test this its kinda sketch
        total=int(totalQubitsInState(state))
        if qubit>total-1: raise ValueError('The measured qubit number cannot be greater than the number of qubits in the state')
        if qubit==0: pa0=GateFunc.P0; pa1=GateFunc.P1
        else:
            pa0=Gate.ID
            for i in range(qubit):
                pa0=nKron(pa,Gate.ID)
            pa1=nKron(pa0,Gate.P1)
            pa0=nKron(pa0,Gate.P0)
        for i in range(total-qubit):
            pa0=nKron(pa0,Gate.ID)
            pa1=nKron(pa1,GAte.ID)
        prob0=np.trace(np.dot(pa0,np.dot(state,state.T)))
        if np.random.rand()<prob0:
            result=0
            newState=normalizeSt(np.dot(pa0,state))
        else:
            result=1
            newState=normalizeSt(np.dot(pa1,state))
        state[0:]=newState
        return result
    
class Register(object):
    #TODO: make this - intialize qubits etc.

class UnitTests(object):
    #TODO: make unit tests - probably one class for each test
    
class algorithms(object):
    #TODO: optionally create some algorithms to run