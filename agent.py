from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json
import torch

def huber_loss(y_true, y_pred, delta=1):
    """Keras implementation for huber loss
    loss = {
        0.5 * (y_true - y_pred)**2 if abs(y_true - y_pred) < delta
        delta * (abs(y_true - y_pred) - 0.5 * delta) otherwise
    }
    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        loss values for all points
    """
    error = (y_true - y_pred)
    abs_error = torch.abs(error)

    quad_error = 0.5*error.pow(2)
    lin_error = delta*(abs_error - 0.5*delta)
    # quadratic error, linear error
    return torch.where(abs_error < delta, quad_error, lin_error)

def mean_huber_loss(y_true, y_pred, delta=1):
    """Calculates the mean value of huber loss

    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        average loss across points
    """
    return huber_loss(y_true, y_pred, delta).mean()

class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    HamiltonianCycleAgent
    BreadthFirstSearchAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col

class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values
    This class extends to the following classes
    PolicyGradientAgent
    AdvantageActorCriticAgent

    Attributes
    ----------
    _model : torch.nn.Module
        The main Q-network used for predictin Q-values
    _target_net : torch.nn.Module
        The target Q-network
    _optimizer : torch.optim.Optimizer
        The optimizer used to train the model
    _loss_fn : torch.nn.modules.loss._Loss
        The loss function used to train the model. Huber loss is used here
    _buffer : ReplayBufferNumpy
        The replay buffer to store experience tuples
    _device : torch.device
        The device to run the model on (cpu or cuda)
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_models()

    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model = self._agent_model().to(self.device)
        if(self._use_target_net):
            self._target_net = self._agent_model().to(self.device)
            self.update_target_net()

        self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr=0.0005)
        self._loss_fn = torch.nn.SmoothL1Loss()  # Huber loss

    def _prepare_input(self, board):
        """Reshape input and normalize
        
        Parameters
        ----------
        board : Numpy array
            The board state to process

        Returns
        -------
        board : Numpy array
            Processed and normalized board
        """
        
        if(board.ndim == 3):
            board = board.reshape((1,) + board.shape)

        board = self._normalize_board(board.copy())
        board = board.transpose((0,3,1,2))  # convert from tensorflows batch,H,W,C to pytorchs batch,C,H,W

        board = torch.from_numpy(board).float()
        board = board.to(self.device)
        return board

    def _get_model_outputs(self, board, model=None):
        """Get action values from the DQN model

        Parameters
        ----------
        board : Numpy array
            The board state for which to predict action values
        model : torch.nn.Module, optional
            The model to use for prediction, if None, use the main model

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board, 
            of shape board.shape[0] * num actions
        """
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model

        model.eval()
        with torch.no_grad():
            outputs = model(board)
        return outputs.cpu().numpy()

    def _normalize_board(self, board):
        """Normalize the board before input to the network
        
        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Numpy array
            The copy of board state after normalization
        """
        # return board.copy()
        # return((board/128.0 - 1).copy())
        return board.astype(np.float32)/4.0

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model)
        return np.argmax(np.where(legal_moves==1, model_outputs, -np.inf), axis=1)

    def _agent_model(self):
        """Returns the model which evaluates Q values for a given state input

        Returns
        -------
        model : torch.nn.Module
            The DQN model architecture
        """
        # define the input layer, shape is dependent on the board size and frames
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            m = json.loads(f.read())

        layers_cfg = m['model'] # dictionary of layers from json
        
        class DQNModel(torch.nn.Module): # A dynamic model class
            ''' Deep Q Network Model that builds itself from json config, dynamically calculates layer sizes and returns the model'''
            def __init__(model_self): # model_self is used to avoid confusion with outer self
                super().__init__()

                layers = [] # list to hold all layers 
                in_channels = self._n_frames
                
                for layer_name,l in layers_cfg.items():
                    # convolutional layer
                    if "Conv2D" in layer_name: 
                        out_channels = l['filters']
                        kernel_size = tuple(l['kernel_size'])
                        strides = tuple(l.get('strides', (1,1)))

                        if l.get('padding', 'valid') == 'same':
                            padding = kernel_size
                        else:
                            padding = 0

                        conv = torch.nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=strides,
                                               padding=padding)
                        layers.append(conv)
                        # relu activation
                        if l.get('activation') == 'relu': 
                            layers.append(torch.nn.ReLU())
                        
                        in_channels = out_channels
                    # flatten layer
                    elif "Flatten" in layer_name: 
                        layers.append(torch.nn.Flatten())
                    # dense layer
                    elif "Dense" in layer_name: 
                        units = l['units']
                        layers.append(torch.nn.LazyLinear(units))

                        if l.get('activation') == 'relu':
                            layers.append(torch.nn.ReLU())
                # final output layer
                layers.append(torch.nn.LazyLinear(self._n_actions)) 
                # sequential model from layers list
                model_self.net = torch.nn.Sequential(*layers) 
            
            def forward(model_self, x):
                ''' Forward pass through the model '''
                return model_self.net(x)
        
        model = DQNModel()
        return model
    
    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk using pytorch save function
        
        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        
        torch.save(
            self._model.state_dict(),
            f"{file_path}/model_{iteration:04d}.pth"
        )

        if self._use_target_net:
            torch.save(
                self._target_net.state_dict(),
                f"{file_path}/model_{iteration:04d}_target.pth"
            )

    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available """
        """Load models from disk using pytorch load function
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0

        main_path = f"{file_path}/model_{iteration:04d}.pth"
        target_path = f"{file_path}/model_{iteration:04d}_target.pth"

        try:
            self._model.load_state_dict(torch.load(main_path, map_location=self.device))
            print(f"Loaded model from {main_path}")
        except FileNotFoundError:
            print(f"Model file not found at {main_path}")

        if self._use_target_net:
            try:
                self._target_net.load_state_dict(torch.load(target_path, map_location=self.device))
                print(f"Loaded target network from {target_path}")
            except FileNotFoundError:
                print(f"Target network file not found at {target_path}")

    def print_models(self):
        """Print the current model architectures using pytorch summary"""
        print('Training Model')
        print(self._model)

        if self._use_target_net:
            print('Target Network')
            print(self._target_net)

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        We are predicting the expected future discounted reward for all
        actions with our model. The target for training the model is calculated
        in two parts:
        1) dicounted reward = current reward + 
                        (max possible reward in next state) * gamma
           the next reward component is calculated using the predictions
           of the target network (for stability)
        2) rewards for only the action take are compared, hence while
           calculating the target, set target value for all other actions
           the same as the model predictions
        
        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the numpy sign command
            rewards > 0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : float
            The current error (error metric is defined in reset_models)
        """
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if(reward_clip):
            r = np.sign(r)

        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model # set the model to use for next state q value prediction
        next_model_outputs = self._get_model_outputs(next_s, current_model) # predict next state q values

        masked_next_q = np.where(legal_moves == 1, next_model_outputs, -np.inf) # mask illegal moves

        max_next_q = np.max(masked_next_q, axis=1).reshape(-1, 1) # max q value for next state

        # our estimate of expexted future discounted reward
        discounted_reward = r + (self._gamma * max_next_q * (1 - done))
        q_values = self._get_model_outputs(s,self._model) # current model outputs
        # we bother only with the difference in reward estimate at the selected action
        target = (1-a)*q_values + a*discounted_reward # Bellman equation to set target values

        # convert to torch tensors
        state_tensor = self._prepare_input(s)
        target_tensor = torch.from_numpy(target).float().to(self.device)

        #forward pass
        self._model.train()
        predicted_q = self._model(state_tensor)
        # fit
        loss = self._loss_fn(predicted_q, target_tensor)

        #Backpropagation
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # loss = round(loss, 5)
        return loss.item()
    
    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())




    # Not used in unsupervised DQN training, useful in supervised pretraining
    def set_weights_trainable(self):
        """Set selected layers to non trainable and compile the model"""
        # freeze all layers
        for param in self._model.parameters():
            param.requires_grad = False
        
        # unfreeze last dense layers. This could be modified to do dynamically but while keras have named layers, pytorch do not
        for idx in [-3, -1]:
            layer = self._model.net[idx]

            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.LazyLinear):
                for param in layer.parameters():
                    param.requires_grad = True
    # Not used in unsupervised DQN training, useful in policy based agents
    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs
    # debugging utility, print whether weights match between model and target net
    def compare_weights(self):
        """Simple utility function to check if the model and target 
        network have the same weights or not
        """

        if not self._use_target_net:
            print("Target network not in use")
            return
        
        model_sd = self._model.state_dict()
        target_sd = self._target_net.state_dict()

        for idx, key in enumerate(model_sd.keys()):
            w_model = model_sd[key]
            w_target = target_sd[key]

            match = int(torch.equal(w_model, w_target))

            print(f"Layer {idx:02d} ({key}) Match : {match}")
    # utility to copy weights from another agent, not in use
    def copy_weights_from_agent(self, agent_for_copy):
        """Update weights between competing agents which can be used
        in parallel training
        """
        assert isinstance(agent_for_copy, type(self)), "Agent type is required for copy"

        self._model.load_state_dict(agent_for_copy._model.state_dict())

        if self._use_target_net and agent_for_copy._use_target_net:
            self._target_net.load_state_dict(agent_for_copy._target_net.state_dict())

# Dummy classes for other agent types
class PolicyGradientAgent():
    def __init__(self):
        pass
class AdvantageActorCriticAgent():
    def __init__(self):
        pass