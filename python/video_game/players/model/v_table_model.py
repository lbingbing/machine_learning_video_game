from . import table_model
from . import v_model
from . import v_table

class VTableModel(table_model.TableModel, v_model.VModel):
    def create_table(self, state):
        return v_table.VTable(state)

    def train(self, batch, learning_rate):
        square_errors = []
        for state, target_V in batch:
            equivalent_state_indexes = state.get_equivalent_state_indexes(state.to_state_index())
            for state_index in equivalent_state_indexes:
                V = self.table.get_V(state_index)
                loss = (V - target_V) ** 2
                dloss = 2 * (V - target_V)
                V -= dloss * learning_rate
                self.table.set_V(state_index, V)
                square_errors.append(loss)
        return sum(square_errors) / len(square_errors)
        
    def get_V(self, state):
        state_index = state.to_state_index()
        V = self.table.get_V(state_index)
        return V
