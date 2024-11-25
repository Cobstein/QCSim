import numpy as np
import scipy as sp
import scipy.linalg
import math

class State(object):
    #TODO: make state seperable? function and seperate state function
    ###ONE QUBIT STATES###
    #z-basis states
    #|0>=(1,0)
    zero=np.array([[1],[0]])
    #|1>=(0,1)
    one=np.array([[0],[1]])
    
    normalize_state=staticmethod(lambda state: state/sp.linalg.norm(state))
    
    #x-basis states
    #|+>=1/sqrt(2)(|0>+|1>)
    plus=normalize_state.__func__(zero+one)
    #|->=1/sqrt(2)(|0>-|1>)
    minus=normalize_state.__func__(zero-one)

    #y-basis states
    #|R>=norm(|0>+i|1>)
    R=normalize_state.__func__(zero+1j*one)
    #|L>=norm(|0>-i|1>)
    L=normalize_state.__func__(zero-1j*one)
    
    @staticmethod
    def nKron(*args):
        result=np.array([[1]])
        for q in args:
            result=np.kron(result,q)
        return result
    
    #Bell state
    cat=normalize_state.__func__(nKron.__func__(zero,zero)+nKron.__func__(one,one))

    total_qubits_in_state=staticmethod(lambda state: math.log2(state.shape[0])/math.log2(2))

class ChangeBases(object):
    #TODO test this more
    @staticmethod
    def change_to_x(state):
        newState=Compute.runGate(Gate.hadamard,state)
        return newState
    
    @staticmethod
    def change_to_y(state):
        newState=Compute.runGate(np.linalg.multi_dot([Gate.hadamard,np.conj(Gate.phaseS.T)]),state)
        return newState
    
    @staticmethod
    def change_to_w(state):
        newState=Compute.runGate(np.linalg.multi_dot([Gate.hadamard,Gate.phaseT,Gate.hadamard,Gate.phaseS]),state)
        return newState
    
    @staticmethod
    def change_to_v(state):
        newState=Compute.runGate(np.linalg.multi_dot([Gate.hadamard,np.conjugate(Gate.phaseT.T),Gate.hadamard,Gate.phaseS]),state)
        return newState
    #TODO: Make change back to Z function - ideally don't need 4


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
    def create_cotrolled_gate(gate,control,target,total):
        #TODO: data validation checks
        if gate.shape!=(2,2):
            raise ValueError('U must be a 2x2 matrix')
        if control<target:
            result=GateFunc.create_cotrolled_gateINT(gate,control,target,total)
        if control>target:
            result=GateFunc.create_cotrolled_gateINT(GateFunc.P1,target,control,total,GateFunc.ID,gate,GateFunc.P0)
        return result
    
    @classmethod
    def create_cotrolled_gateINT(self,gate,control,target,total,P0=P0,P1=P1,ID=ID):
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
    Rx=staticmethod(lambda shift: sp.linalg.exp(-1j*Gate.pauliX*shift/2))
    Ry=staticmethod(lambda shift: sp.linalg.exp(-1j*Gate.pauliY*shift/2))
    Rz=staticmethod(lambda shift: sp.linalg.exp(-1j*Gate.pauliZ*shift/2))

    #TODO: Programatically create CCNOT gates and the like
    
    #Common Controlled Gates upt to 5 qubits
    CNOT=dict()
    for total in range(2,6):
        for control in range(5):
            if control<total:
                for target in range(5):
                    if target<total and control!=target:
                        CNOT[control,target,total]=GateFunc.create_cotrolled_gate(NOT,control,target,total)
    
    cPhaseS=dict()
    for total in range(2,6):
        for control in range(5):
            if control<total:
                for target in range(5):
                    if target<total and control!=target:
                        cPhaseS[control,target,total]=GateFunc.create_cotrolled_gate(phaseS,control,target,total)
    cPhase=dict()
    @staticmethod
    def c_phase(shift):
        for total in range(2,6):
            for control in range(5):
                if control<total:
                    for target in range(5):
                        if target<total and control!=target:
                            Gate.cPhase[control,target,total,shift]=GateFunc.create_cotrolled_gate(Gate.gPhase(shift),control,target,total)
        return Gate.cPhase

    #Swap gate
    #TODO: make this programatically
    SWAP=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])



class Compute(object):
    #TODO: actually make this a computer
    @staticmethod
    def run_gate(gate,state):
        if gate.shape[0]==state.shape[0]:
            newState=np.dot(gate,state)
        else:
            raise ValueError('gate size does not match total qubits in state')
        return newState
        


class Measure(object):
    @staticmethod
    def measure(state,qubit):
        #TODO: test this its kinda sketch
        #TODO: this won't work for entangled qubits
        total=int(State.total_qubits_in_state(state))
        if qubit>total-1: raise ValueError('The measured qubit number cannot be greater than the number of qubits in the state')
        if qubit==0: pa0=GateFunc.P0; pa1=GateFunc.P1
        else:
            pa0=Gate.ID
            for i in range(qubit):
                pa0=State.nKron(pa0,Gate.ID)
            pa1=State.nKron(pa0,Gate.P1)
            pa0=State.nKron(pa0,Gate.P0)
        for i in range(total-qubit):
            pa0=State.nKron(pa0,Gate.ID)
            pa1=State.nKron(pa1,Gate.ID)
        prob0=np.trace(np.dot(pa0,np.dot(state,state.T)))
        if np.random.rand()<prob0:
            result=0
            newState=State.normalize_state(np.dot(pa0,state))
        else:
            result=1
            newState=State.normalize_state(np.dot(pa1,state))
        state[0:]=newState
        return result
    
class Register(object):
    #TODO: make this - intialize qubits etc.
    #This will be user input
    #example input 
    def __init__(self,state=State.zero,entangled=None):
        self._entangled=[self]
        self._state=state
        self._noop = [] #TODO: after a measurement set this so that we can allow no further operations. Set to Bloch coords if bloch operation performed
    def get_entangled(self):
        return self._entangled
    def set_entangled(self,entangled):
        self._entangled=entangled
        for q in self._entangled:
            q._state=self._state
            q._entangled=self._entangled
    def get_state(self):
        return self._state
    def set_state(self,state):
        self._state=state
        for q in self._entangled:
            q._state=state
            q._entangled=self._entangled
            q._noop=self._noop
    def get_noop(self):
        return self._noop
    def set_noop(self,noop):
        self._noop=noop
        for q in self._entangled:
            q._noop=noop
    def is_entangled_state(self):
        return len(self._entangled)>1
    def is_entangled_with(self,qubit):
        return qubit in self._entangled
    def num_qubits(self):
        return State.total_qubits_in_state(self._state)
    


class UnitTests(object):
    #TODO: make unit tests - probably one class for each test
    p=1
class Algorithms(object):
    #TODO: optionally create some algorithms to run
    p=1