from criteria.partial_fc import PartialFC_V2, CombinedMarginLoss
from pytorch_metric_learning import losses
from torch import distributed

def select(loss, config):
    if loss == 'arcface':
        return losses.ArcFaceLoss(num_classes=config.n_classes,
                                  embedding_size=config.embedding_size,
                                  margin=28.6,
                                  scale=64)      
    if loss == 'angular':
        return losses.AngularLoss(alpha=40)
    if loss == 'tripletmargin':
        return losses.TripletMarginLoss(margin=0.05,
                                        swap=False,
                                        smooth_loss=False,
                                        triplets_per_anchor="all")   
    if loss == 'multisimilarity':
        return losses.MultiSimilarityLoss(alpha=2,
                                          beta=50,
                                          base=0.5)   
    if loss == 'margin':
        return losses.MarginLoss(num_classes=config.n_classes,
                                 margin=0.2, 
                                 nu=0, 
                                 beta=1.2, 
                                 triplets_per_anchor="all", 
                                 learn_beta=False) 
    if loss == 'partialfc':
        world_size = 1
        rank = 0
        distributed.init_process_group(
            backend="gloo",  # gloo on windows / nccl on linux
            init_method="tcp://127.0.0.1:12584",
            rank=rank,
            world_size=world_size,
        )
        margin_loss = CombinedMarginLoss(64,1,0.5,0,0)
        return PartialFC_V2(margin_loss=margin_loss,
                            embedding_size=config.embedding_size,
                            num_classes=config.n_classes,
                            sample_rate=0.2)   
    
    raise NotImplementedError('Loss {} not implemented!'.format(loss))