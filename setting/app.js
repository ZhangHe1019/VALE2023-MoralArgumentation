import React, { useState } from 'react';
import { Streamlit, withStreamlitConnection } from 'streamlit-component-lib';
import DropdownTreeSelect from 'react-dropdown-tree-select';

const App = ({ args }) => {
  const [selectedValue, setSelectedValue] = useState(null);

  return (
    <div>
      <DropdownTreeSelect
        data={args.data}
        onChange={(currentNode, selectedNodes) => {
          setSelectedValue(selectedNodes);
          Streamlit.setComponentValue(selectedNodes);
        }}
      />
    </div>
  );
}

export default withStreamlitConnection(App);
