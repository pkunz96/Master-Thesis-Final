from __future__ import annotations
from typing import List, Dict, Tuple
from enum import Enum

class AddressContext(Enum):

    REGISTER_FILE = 0

    STACK_REGISTER_FILE = 1

    IN_ARR = 2

    OUT_ARR = 3

    HELPER_ARR = 4

class Executor:

    def execute(self, instruction: str) -> Executor:
        pass

    def get_instruction_set(self) -> List[str]:
        pass

class Stack(Executor):

    def __init__(self, environment: Environment):
        self.stack: List[int] = []
        self.environment = environment

    def push_constant(self, constant: int) -> Stack:
        self.environment.update_pass_record(20 + constant, True, -1)
        self.stack.append(constant)
        return self

    def push_address_type(self, address_type: int) -> Stack:
        if address_type < 0 or address_type > 4:
            raise ValueError()
        self.environment.update_pass_record(address_type, True, -1)
        self.stack.append(address_type)
        return self

    def push_input_size(self) -> Stack:
        self.environment.update_pass_record(5, True, -1)
        self.stack.append(self.environment.in_arr_size)
        return self

    def push_in_arr_length(self) -> Stack:
        self.environment.update_pass_record(6, True, -1)
        self.stack.append(len(self.environment.in_arr))
        return self

    def push_out_arr_length(self) -> Stack:
        self.environment.update_pass_record(7, True, -1)
        self.stack.append((len(self.environment.out_arr)))
        return self

    def push_helper_arr_length(self) -> Stack:
        self.environment.update_pass_record(8, True, -1)
        self.stack.append(len(self.environment.helper_arr))
        return self

    def extend_in_arr(self) -> Stack:
        if len(self.stack) > 0:
            self.environment.update_pass_record(9, True, -1)
            num_additional_elements: int = self.stack.pop()
            self.environment.in_arr = [0] * num_additional_elements + self.environment.in_arr
        else:
            raise RuntimeError()
        return self

    def duplicate(self) -> Stack:
        self.environment.update_pass_record(10, True, -1)
        self.environment.in_arr = 2 * self.environment.in_arr
        return self

    def push(self) -> Stack:
        if len(self.stack) >= 2:
            address: int = self.stack.pop()
            address_type: int = self.stack.pop()
            if address_type < 0 or address_type > 4:
                raise ValueError()
            else:
                self.environment.update_pass_record(11, True, -1)
                value: int
                if address_type == AddressContext.REGISTER_FILE.value:
                    value = self.environment.register_file.register_file[address]
                elif address_type == AddressContext.STACK_REGISTER_FILE.value:
                    stack_register_file_top_index: int = len(self.environment.stack_register_file.register_file_stack) - 1
                    value = self.environment.stack_register_file.register_file_stack[stack_register_file_top_index][address]
                elif address_type == AddressContext.IN_ARR.value:
                    value = self.environment.in_arr[address]
                elif address_type == AddressContext.HELPER_ARR.value:
                    value = self.environment.helper_arr[address]
                else:
                    value = self.environment.out_arr[address]
                self.stack.append(value)
        else:
            raise RuntimeError()
        return self;

    def store(self) -> Stack:
        value: int = self.stack.pop()
        address: int = self.stack.pop()
        address_type: int = self.stack.pop()
        if address_type < 0 or address_type > 4:
            raise ValueError()
        else:
            self.environment.update_pass_record(12, True, -1)
            if address_type == AddressContext.REGISTER_FILE.value:
                self.environment.register_file.register_file[address] = value
            elif address_type == AddressContext.STACK_REGISTER_FILE.value:
                stack_register_file_top_index: int = len(self.environment.stack_register_file.register_file_stack) - 1
                self.environment.stack_register_file.register_file_stack[stack_register_file_top_index][address]  = value
            elif address_type == AddressContext.IN_ARR.value:
                self.environment.in_arr[address] = value
            elif address_type == AddressContext.HELPER_ARR.value:
                self.environment.helper_arr[address] = value
            elif address_type == AddressContext.OUT_ARR.value:
                self.environment.out_arr[address] = value
        return self

    def add(self) -> Stack:
        self.environment.update_pass_record(13, True, -1)
        x = self.stack.pop()
        y = self.stack.pop()
        self.stack.append(y + x)
        return self

    def sub(self) -> Stack:
        self.environment.update_pass_record(14, True, -1)
        x = self.stack.pop()
        y = self.stack.pop()
        self.stack.append(y - x)
        return self

    def div(self) -> Stack:
        self.environment.update_pass_record(15, True, -1)
        x = self.stack.pop()
        y = self.stack.pop()
        self.stack.append(y // x)
        return self

    def mult(self) -> Stack:
        self.environment.update_pass_record(16, True, -1)
        x = self.stack.pop()
        y = self.stack.pop()
        self.stack.append(x*y)
        return self

    def square(self) -> Stack:
        self.environment.update_pass_record(17, True, -1)
        x = self.stack.pop()
        self.stack.append(x**2)
        return self


class RegisterFile(Executor):

    @staticmethod
    def create_register_file(size: int) -> Dict[int, int]:
        return dict.fromkeys(list(range(0, size)), 0)

    def __init__(self, environment: Environment, width: int):
        self.environment = environment
        self.register_file = RegisterFile.create_register_file(width)

    def get_value(self, address: int) -> int:
        return self.register_file[address]


class RegisterFileStack(Executor):

    def __init__(self, environment: Environment, width: int):
        self.environment = environment
        self.width = width
        self.register_file_stack = [RegisterFile.create_register_file(width)]

    def get_value(self, address: int) -> int:
        return self.register_file_stack[len(self.register_file_stack) - 1][address];

    def push_register_file_stack(self):
        self.environment.update_pass_record(18, True, -1)
        self.register_file_stack.append(RegisterFile.create_register_file(self.width))
        return self

    def pop_register_file_stack(self):
        if len(self.register_file_stack) >= 2:
            self.environment.update_pass_record(19, True, -1)
            self.register_file_stack.pop()
        else:
            raise RuntimeError();


class Environment(Executor):

    def __init__(self, in_arr: List[int], out_arr: List[str],  helper_arr: List[str], register_file_width: int, stack_register_file_width: int, include_stack_length: bool = False):
        # Components
        self.stack = Stack(self)
        self.register_file = RegisterFile(self, register_file_width)
        self.stack_register_file = RegisterFileStack(self, stack_register_file_width)
        # Data Structures
        self.in_arr = in_arr
        self.helper_arr = helper_arr
        self.out_arr = out_arr
        # Meta Information
        self.in_arr_size = len(in_arr)
        self.max_constant_value = max([2, register_file_width, stack_register_file_width])
        # Management Data Structures
        self.binding_dict = dict()
        self.reg_bindings = []
        self.arr_bindings = []
        self.binding_arr_address_type = dict()
        self.next_binding_id = 0
        self.cur_pc = 0
        self.pass_record = []
        self.include_stack_length = include_stack_length

    def bind_reg(self, address: int, address_type: int) -> int:
        if address_type < 0 or address_type > 1:
            raise ValueError()
        binding_id = self.next_binding_id
        self.next_binding_id += 1
        self.binding_dict[binding_id] = (address_type, address)
        self.reg_bindings.append(binding_id)
        return binding_id

    def value_reg(self, binding_id: int) -> int:
        if binding_id not in self.binding_dict:
            raise ValueError()
        if binding_id in self.binding_dict:
            address_type, address = self.binding_dict[binding_id]
            value: int
            if address_type == AddressContext.REGISTER_FILE.value:
                value = self.register_file.register_file[address]
            elif address_type == AddressContext.STACK_REGISTER_FILE.value:
                stack_register_file_top_index: int = len(self.stack_register_file.register_file_stack) - 1
                value = self.stack_register_file.register_file_stack[stack_register_file_top_index][address]
            else:
                raise RuntimeError()
        else:
            raise ValueError()
        return value

    def bind_arr(self, address: int, address_type: int, arr_address_type: int) -> int:
        if address_type < 0 or address_type > 1:
            raise ValueError()
        binding_id = self.bind_reg(address, address_type)
        self.binding_arr_address_type[binding_id] = arr_address_type
        self.arr_bindings.append(binding_id)
        return binding_id

    def value_arr(self, binding_id: int) -> int:
        address = self.value_reg(binding_id);
        address_type = self.binding_arr_address_type[binding_id]
        value: int = 0
        if address_type == AddressContext.IN_ARR.value:
            if address < len(self.in_arr):
                value = self.in_arr[address]
        elif address_type == AddressContext.OUT_ARR.value:
            if address < len(self.out_arr):
                value = self.out_arr[address]
        elif address_type == AddressContext.HELPER_ARR.value:
            if address < len(self.helper_arr):
                value = self.helper_arr[address]
        else:
            RuntimeError()
        return value

    def update_pass_record(self, instruction: int, increment_pc: bool, new_pc: int) -> None:
        cur_pc: int = self.cur_pc
        if increment_pc:
            self.cur_pc += 1
        else:
            self.cur_pc = new_pc
        vector = tuple()
        vector += tuple(self.register_file.register_file.values())
        vector += tuple(self.stack_register_file.register_file_stack[len(self.stack_register_file.register_file_stack) - 1].values())
        for key in self.arr_bindings:
            if key in self.arr_bindings:
                vector += (self.value_arr(key),)
        if self.include_stack_length:
            vector += (len(self.stack_register_file.register_file_stack),)
        # vector += (cur_pc, instruction, -1 + int(increment_pc) * 2, new_pc)
        # We aim to predict the next state given the current state (cur_pc). Therefore, the current instruction is
        # irrelevant and excluded.
        vector += (cur_pc,)
        self.pass_record.append(vector)

    def as_vector(self) -> Tuple[int, ...]:
        return self.pass_record[len(self.pass_record) - 1]

    def set_program_counter(self, new_pc: int) -> Executor:
        self.update_pass_record(-1, False, new_pc)
        return self


def name_for_operation(record: Tuple[int, ...]) -> str:
    id = record[len(record) - 3]
    if id == -1:
        return "IDENTITY"
    elif 0 <= id <= 4:
        return "PUSH_ADDRESS_TYPE_" + str(id)
    elif id == 5:
        return "PUSH_INPUT_SIZE"
    elif id == 6:
        return "PUSH_IN_ARR_LENGTH"
    elif id == 7:
        return "PUSH_OUT_ARR_LENGTH"
    elif id == 8:
        return "PUSH_HELPER_ARR_LENGTH"
    elif id == 9:
        return "EXTEND_IN_ARR"
    elif id == 10:
        return "DUPLICATE"
    elif id == 11:
        return "PUSH"
    elif id == 12:
        return "STORE"
    elif id == 13:
        return "ADD"
    elif id == 14:
        return "SUB"
    elif id == 15:
        return "DIV"
    elif id == 16:
        return "MULT"
    elif id == 17:
        return "SQUARE"
    elif id == 18:
        return "PUSH_REGISTER_FILE"
    elif id == 19:
        return "POP_REGISTER_FILE"
    else:
        return "PUSH_CONSTANT_" + str(id-20)


def name_for_transition(record: Tuple[int, ...]) -> str:
    next_state = record[-2:]
    if next_state == (1, -1):
        return "INCREMENT"
    elif next_state[0] == -1 and next_state[1] >= 0:
        return "SET_TO_" + str(next_state[1])
    else:
        raise ValueError()

