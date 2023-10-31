"""
 Functions to compute the computational complexities of CVNNs
 
 CVFNN  - cvfnn_complexity
 SCFNN  - scfnn_complexity
 MLMVN  - mlmvn_complexity
 C-RBF  - crbf_complexity
 FC-RBF - fcrbf_complexity
 PT-RBF - ptrbf_complexity

 Created by Kayol Soares Mayer
 Last updated 2023/10/31

 Cite:
 @INPROCEEDINGS{Mayer2023,
  author={Mayer, Kayol Soares and and Soares, 
          Jonathan Aguiar and Cruz, Ariadne Arrais and 
           Arantes, Dalton Soares},
  booktitle={IEEE Latin-American Conference on Communications}, 
  title={On the computational complexities of complex-valued neural networks}, 
  year={2023},
  pages = {1--5},
  }
"""

def cvfnn_complexity(I):
    """
    CVFNN computational complexity (number of real-valued multiplications).
        It only considers the SGD optimizer.
    
    Example for a CVFNN with 6 inputs, 97 hidden neurons, and 3 outputs.
    >>> flop_training, flop_inference = cvfnn_complexity([6, 97, 3])   
    >>> flop_training = 8948
    >>> flop_inference = 3492    

    Parameters
    ----------
        :I (list): Number of neurons per layer, including input and output 
            layers.

    Returns
    -------
        :t (int): FLOP per training iteration.
        :i (int): FLOP per inference iteration.
    """
    
    # Training complexity
    t = 4*sum([I[l]*(2*I[l-1]+I[l+1]+2) for l in range(1,len(I)-1)])\
                                                             +8*I[-1]*(I[-2]+1)
                                                             
    # inference complexity
    i = 4*sum([I[l]*I[l-1] for l in range(1,len(I))])
                                                             
    
    return t, i
      

def scfnn_complexity(I):
    """
    SCFNN computational complexity (number of real-valued multiplications).
        It only considers the SGD optimizer.
    
    Example for a SCFNN with 6 inputs, 97 hidden neurons, and 3 outputs.
    >>> flop_training, flop_inference = scfnn_complexity([6, 97, 3])   
    >>> flop_training = 8942
    >>> flop_inference = 3492    

    Parameters
    ----------
        :I (list): Number of neurons per layer, including input and output 
            layers.

    Returns
    -------
        :t (int): FLOP per training iteration.
        :i (int): FLOP per inference iteration.
    """
    
    # Training complexity
    t = 4*sum([I[l]*(2*I[l-1]+I[l+1]+2) for l in range(1,len(I)-1)])\
                                                           +2*I[-1]*(4*I[-2]+3)
                                                             
    # inference complexity
    i = 4*sum([I[l]*I[l-1] for l in range(1,len(I))])
                                                             
    
    return t, i


def mlmvn_complexity(I):
    """
    MLMVN computational complexity (number of real-valued multiplications).
        It only considers the SGD optimizer.
    
    Example for a MLMVN with 6 inputs, 97 hidden neurons, and 3 outputs.
    >>> flop_training, flop_inference = mlmvn_complexity([6, 97, 3])   
    >>> flop_training = 9736
    >>> flop_inference = 3892    

    Parameters
    ----------
        :I (list): Number of neurons per layer, including input and output 
            layers.

    Returns
    -------
        :t (int): FLOP per training iteration.
        :i (int): FLOP per inference iteration.
    """
    
    # Training complexity
    t = 4*sum([I[l]*(2*I[l-1]+I[l+1]+4) for l in range(1,len(I)-1)])\
                                                           +4*I[-1]*(2*I[-2]+3)
                                                             
    # inference complexity
    i = 4*sum([I[l]*(I[l-1]+1) for l in range(1,len(I))])
                                                             
    
    return t, i


def crbf_complexity(inp, neurons, out):
    """
    C-RBF computational complexity (number of real-valued multiplications).
        It only considers the SGD optimizer.
    
    Example for a C-RBF with 6 inputs, 100 hidden neurons, and 3 outputs. This
        CVNN is only available in shallow architectures in the literature.
    >>> flop_training, flop_inference = crbf_complexity(6, 100, 3)   
    >>> flop_training = 4712
    >>> flop_inference = 1900    

    Parameters
    ----------
        :inp (int): Number of inputs.
        :neurons (int): Number of neurons in its unique layer with neurons.
        :out (int): Number of outputs.

    Returns
    -------
        :t (int): FLOP per training iteration.
        :i (int): FLOP per inference iteration.
    """
    
    # Training complexity
    t = neurons*(4*inp+6*out+5)+4*out
                                                             
    # inference complexity
    i = neurons*(2*inp+2*out+1)
                                                             
    
    return t, i


def fcrbf_complexity(inp, neurons, out):
    """
    FC-RBF computational complexity (number of real-valued multiplications).
        It only considers the SGD optimizer.
    
    Example for a FC-RBF with 6 inputs, 100 hidden neurons, and 3 outputs. This
        CVNN is only available in shallow architectures in the literature.
    >>> flop_training, flop_inference = fcrbf_complexity(6, 100, 3)   
    >>> flop_training = 12012
    >>> flop_inference = 3600    

    Parameters
    ----------
        :inp (int): Number of inputs.
        :neurons (int): Number of neurons in its unique layer with neurons.
        :out (int): Number of outputs.

    Returns
    -------
        :t (int): FLOP per training iteration.
        :i (int): FLOP per inference iteration.
    """
    
    # Training complexity
    t = 12*neurons*(inp+out+1)+4*out
                                                             
    # inference complexity
    i = 4*neurons*(inp+out)
                                                             
    
    return t, i


def ptrbf_complexity(I, O):
    """
    PT-RBF computational complexity (number of real-valued multiplications).
        It only considers the SGD optimizer.
    
    Example for a PT-RBF with 6 inputs, 50 neurons and outputs in the first 
        hidden layer, and 50 neurons and 3 outputs in the second hidden layer.
    >>> flop_training, flop_inference = ptrbf_complexity([50, 50],[6, 50, 3])   
    >>> flop_training = 54412
    >>> flop_inference = 16400    

    Parameters
    ----------
        :I (list): Number of neurons per layer.
        :O (list): Number of outputs per layer, including input and output 
            layers.

    Returns
    -------
        :t (int): FLOP per training iteration.
        :i (int): FLOP per inference iteration.
    """
    
    # Append 0 to the beggining of I to avoid access issues
    I = [0] + I
    
    # Training complexity
    t = 4*sum([I[l]*(O[l-1]+3*O[l]+3) for l in range(1,len(I))])\
                  + 4*sum([O[l]*(I[l+1]+1) for l in range(1,len(I)-1)])+4*O[-1]
                                                             
    # inference complexity
    i = 2*sum([I[l]*(O[l-1]+2*O[l]+1) for l in range(1,len(I))])
                                                             
    
    return t, i






