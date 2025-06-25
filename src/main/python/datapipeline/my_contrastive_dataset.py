# src/main/python/datapipeline/my_contrastive_dataset.py
import torch.utils.data as data
import random

class MyContrastiveDataset(data.Dataset):
    """
    Creates training instances by sampling from pre-computed positive/negative lists.
    Uses precise character indices to insert entity markers.
    """
    def __init__(self, data_loader, num_negatives: int = 4):
        super().__init__()
        self.df = data_loader.load()
        self.num_negatives = num_negatives
        self.level_map = {
            'pos_level_1': 1, 'pos_level_2': 2, 'pos_level_3': 3,
            'neg_level_1': 4, 'neg_level_2': 5, 'neg_level_3': 6
        }
        self.positive_cols = ['pos_level_1', 'pos_level_2', 'pos_level_3']
        self.negative_cols = ['neg_level_1', 'neg_level_2', 'neg_level_3']

    def __len__(self) -> int:
        return len(self.df)

    def _format_sentence(self, row, mask_token: str):
        # This function robustly inserts entity markers using character indices.
        text = row['sent']
        
        # Define markers
        e1_markers = ("[E1]", "[/E1]")
        e2_markers = ("[E2]", "[/E2]")
        
        # Get entity positions
        e1_start, e1_end = row['e1_start_pos'], row['e1_end_pos']
        e2_start, e2_end = row['e2_start_pos'], row['e2_end_pos']
        
        # Order the replacements to avoid messing up indices
        if e1_start < e2_start:
            # E1 comes first
            marked_text = (
                text[:e1_start] +
                f" {e1_markers[0]} " + text[e1_start:e1_end] + f" {e1_markers[1]} " +
                text[e1_end:e2_start] +
                f" {e2_markers[0]} " + text[e2_start:e2_end] + f" {e2_markers[1]} " +
                text[e2_end:]
            )
        else:
            # E2 comes first
            marked_text = (
                text[:e2_start] +
                f" {e2_markers[0]} " + text[e2_start:e2_end] + f" {e2_markers[1]} " +
                text[e2_end:e1_start] +
                f" {e1_markers[0]} " + text[e1_start:e1_end] + f" {e1_markers[1]} " +
                text[e1_end:]
            )
            
        # Append the relation prompt from the original repository's strategy
        final_text = f"{marked_text.strip()} The relation is {mask_token}."
        return final_text


    def __getitem__(self, index: int):
        # The [MASK] token will be passed in from the collate function, which has the tokenizer.
        # This is a placeholder for now, we'll get the real one in the collate_fn.
        mask_token = "[MASK]" 
        anchor_row = self.df.iloc[index]
        
        anchor_text = self._format_sentence(anchor_row, mask_token)
        
        candidates_text = []
        candidates_levels = []

        # Sample one positive
        available_pos = [col for col in self.positive_cols if len(anchor_row[col]) > 0]
        if available_pos:
            chosen_pos_col = random.choice(available_pos)
            pos_idx = random.choice(anchor_row[chosen_pos_col])
            pos_row = self.df.iloc[pos_idx]
            candidates_text.append(self._format_sentence(pos_row, mask_token))
            candidates_levels.append(self.level_map[chosen_pos_col])
        
        # To ensure we always have at least one positive, let's add the anchor itself as a hard positive
        # if no other positive is found. This is a common practice.
        if not candidates_text:
             candidates_text.append(self._format_sentence(anchor_row, mask_token))
             candidates_levels.append(1) # Treat self as hard positive

        # Sample N negatives
        available_neg = [col for col in self.negative_cols if len(anchor_row[col]) > 0]
        for _ in range(self.num_negatives):
             if available_neg:
                chosen_neg_col = random.choice(available_neg)
                neg_idx = random.choice(anchor_row[chosen_neg_col])
                neg_row = self.df.iloc[neg_idx]
                candidates_text.append(self._format_sentence(neg_row, mask_token))
                candidates_levels.append(self.level_map[chosen_neg_col])

        return anchor_text, candidates_text, candidates_levels