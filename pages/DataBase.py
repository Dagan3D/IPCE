import streamlit as st
from sqlalchemy.sql import text
import datetime
conn = st.connection('pets_db', type='sql')


with conn.session as s:
    if st.button("Create table"):
        st.markdown(f"Note that `s` is a `{type(s)}`")
        s.execute(text('CREATE TABLE IF NOT EXISTS pet_owners (person TEXT, pet TEXT);'))
        s.execute(text('DELETE FROM pet_owners;'))
        pet_owners = {'jerry': 'fish', 'barbara': 'cat', 'alex': 'puppy'}
        for k in pet_owners:
            s.execute(
                text('INSERT INTO pet_owners (person, pet) VALUES (:owner, :pet);'),
                params=dict(owner=k, pet=pet_owners[k])
            )
        s.commit()

with conn.session as s:
    if st.button("Add row"):
        s.execute(
            text('INSERT INTO pet_owners (person, pet) VALUES ("random", "pet");')
        )
        s.commit()

with conn.session as s:
    df = conn.query("select * from pet_owners", ttl=0.5)
    st.dataframe(df)

    

