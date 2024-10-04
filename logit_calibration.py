import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Ensure true_labels is of type long
        true_labels = true_labels.long()
        
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
    # Test Loca
        # Test 2 method of logit calibration for teacher model
    # set random seed
    torch.manual_seed(0)
    calibrated_fn = Loca(0.9)
    teacher_logits = torch.randn(10, 10)
    true_labels = torch.randint(0, 10, (10,))
    student_logits  = torch.randn(10, 10)

    calibrated_logits= calibrated_fn(teacher_logits, true_labels)
    # print(temp/)
    print(calibrated_logits)
    # calculate the loss using kl divergence between student and 
    # use the temp to change softmax temperature
    loss = 0
    # for i in range(10):
    #     loss += F.kl_div(F.log_softmax(student_logits[i]/temp[i], dim=0), F.softmax(calibrated_logits[i]/temp[i], dim=0), reduction='batchmean')
    #     print(loss)

    # loss = F.kl_div(F.log_softmax(student_logits/temp, dim=1), F.softmax(calibrated_logits/temp, dim=1), reduction='batchmean')
    print(loss)
    new_loss = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(calibrated_logits, dim=1), reduction='batchmean')
    print(new_loss)