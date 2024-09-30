import torch
import torch.nn as nn


# Define the logit calibration function
# Does this run in GPU?
# Note from 27/08/2024
# Try with alpha = 1 with true label teacher logit will remaining the same
# W
class Loca(nn.Module):
    def __init__(self, alpha):
        """
        Calibrate the teacher logits based on the true labels.
        
        Args:
            alpha (float): Scaling factor for the calibration.
            paper note alpha go from 0.9 to 1.0
        """
        super(Loca, self).__init__()
        self.alpha = alpha
    
    def forward(self, teacher_logits, true_labels):
        # Get the number of classes
        num_classes = teacher_logits.size(1)
        
        # Create a tensor for calibrated logits
        calibrated_logits = torch.zeros_like(teacher_logits)
        
        # Iterate over each sample
        for i in range(teacher_logits.size(0)):
            # Get the true label for the current sample
            true_label = true_labels[i]
            
            # Calculate the scaling factor
            s = self.alpha * (1 / (1 - teacher_logits[i, true_label].item() + 
                                  torch.sum(teacher_logits[i, :]) - teacher_logits[i, true_label].item()))
            
            # Adjust the logits
            for j in range(num_classes):
                if j == true_label:
                    calibrated_logits[i, j] = 1 - s * torch.sum(teacher_logits[i, :]) + s * teacher_logits[i, j]
                else:
                    calibrated_logits[i, j] = s * teacher_logits[i, j]
        
        return calibrated_logits


class LogitCalibration2(nn.Module):
    def __init__(self, temp):
        super(LogitCalibration2, self).__init__()
        """
        # When predicted label is same as true label, use the teacher logit with softmax temp = temp
        # When predicted label is wrong, use the teacher logit with softmax temp = wrong_teacher_temp
        
        Calibrate the teacher logits by setting the logit of the true label to 1 and the rest to 0.
        
        Args:
            teacher_logits (torch.Tensor): Logits from the teacher model.
            true_labels (torch.Tensor): True labels for the samples.
        
        Returns:
            torch.Tensor: Calibrated logits.
            torch.Tensor: teachertemp
        """
        self.temp = temp

    def forward(self, teacher_logits, true_labels):

        # Get the number of classes
        num_classes = teacher_logits.size(1)
        # a Tensor hold position value of temp for right and wrong teacher logit with size of batch
        teachertemp = torch.where(torch.argmax(teacher_logits, dim=1) == true_labels, self.temp, 1.0)
        # Create a tensor for calibrated logits
        calibrated_logits = torch.zeros_like(teacher_logits)
        
        # Iterate over each sample
        for i in range(teacher_logits.size(0)):
            # Get the true label for the current sample
            true_label = true_labels[i]
            
            # Get the predicted label from the teacher model
            predicted_label = torch.argmax(teacher_logits[i, :])
            
            # If the predicted label is the same as the true label, use the teacher logit
            if predicted_label == true_label:
                calibrated_logits[i, :] = teacher_logits[i, :]
            # Else, set the logit of the true label to 1 and the rest to 0 
            else:
                calibrated_logits[i, true_label] = 1
                for j in range(num_classes):
                    if j != true_label:
                        calibrated_logits[i, j] = 0
        
        return calibrated_logits, teachertemp

if __name__ == '__main__':
    pass