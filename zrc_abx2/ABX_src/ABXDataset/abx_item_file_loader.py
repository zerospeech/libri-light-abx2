from ..models import *

# ITEMFILE COLUMNS
FILEID_COL = 0
ONSET_COL = 1
OFFSET_COL = 2
PHONE_COL = 3
PREV_PHONE_COL = 4
NXT_PHONE_COL = 5
SPEAKER_COL = 6
COLUMN_COUNT = 7

# ITEMDATA_INDICES
# These are the indices for the constructed itemdata
# This assumes onset, offset, context_id, phone_id, speaker_id
ONSET_IDX = 0
OFFSET_IDX = 1
CONTEXT_IDX = 2
PHONE_IDX = 3
SPEAKER_IDX = 4


class ABXItemFileLoader:
    def load_item_file(self, path_item_file: str) -> ItemFile:
        r"""Load a .item file indicating the triplets for the ABX score. The
        input file must have the following format:
        line 0 : whatever (not read)
        line > 0: #file_ID onset offset #phone prev-phone next-phone speaker
        onset : begining of the triplet (in s)
        onset : end of the triplet (in s)

        Returns a tuple of files_data, context_match, phone_match, speaker_match where
                files_data: dictionary whose key is the file id, and the value is the list of item tokens in that file, each item in turn
                                given as a list of onset, offset, context_id, phone_id, speaker_id.
                context_match is a dictionary of the form { prev_phone_str+next_phone_str: context_id }.
                phone_match is a dictionary of the form { phone_str: phone_id }.
                speaker_match is a dictionary of the form { speaker_str: speaker_id }.
                The id in each case is iterative (0, 1 ...)
        """
        with open(path_item_file, "r") as file:
            item_f_lines = file.readlines()[1:]

        item_f_lines = [x.replace("\n", "") for x in item_f_lines]

        # key: fileID, value: a list of items, each item in turn given as a list of
        # onset, offset, context_id, phone_id, speaker_id (see below for the id constructions)
        files_data: dict[str, list[ItemData]] = {}

        # Provide a phone_id for each phoneme type (a la B: 0, N: 1 ...)
        phone_match: dict[str, int] = {}
        context_match: dict[str, int] = {}  # ... context_id ...
        speaker_match: dict[str, int] = {}  # ... speaker_id ...

        for line in item_f_lines:
            items = line.split()
            assert len(items) == COLUMN_COUNT  # assumes 7-column files
            fileID = items[FILEID_COL]
            if fileID not in files_data:
                files_data[fileID] = []

            onset, offset = float(items[ONSET_COL]), float(items[OFFSET_COL])
            phone = items[PHONE_COL]

            speaker = items[SPEAKER_COL]
            context = "+".join([items[PREV_PHONE_COL], items[NXT_PHONE_COL]])

            if phone not in phone_match:
                # We increment the id by 1 each time a new phoneme type is found
                s = len(phone_match)
                phone_match[phone] = s
            phone_id = phone_match[phone]

            if context not in context_match:
                s = len(context_match)
                context_match[context] = s
            context_id = context_match[context]

            if speaker not in speaker_match:
                s = len(speaker_match)
                speaker_match[speaker] = s
            speaker_id = speaker_match[speaker]

            files_data[fileID].append(
                ItemData(onset, offset, context_id, phone_id, speaker_id)
            )

        return ItemFile(files_data, context_match, phone_match, speaker_match)
