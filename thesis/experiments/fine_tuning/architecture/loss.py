from deeplearning.losses.torch import MSELoss


class DMSTLoss(MSELoss):

    def __init__(self, 
        scale,
        **kwargs):
        super().__init__(**kwargs)

        self.scale = scale

    def forward(self, x, _):

        new_noise, neutral_noise, positive_noise = x

        target = neutral_noise - (self.scale * (positive_noise - neutral_noise))

        negative_loss = super().forward(new_noise, target)

        return negative_loss

    @staticmethod
    def args(parser):

        parser.add_argument('--scale', type=float, default=1.0)

        super(DMSTLoss, DMSTLoss).args(parser)


    