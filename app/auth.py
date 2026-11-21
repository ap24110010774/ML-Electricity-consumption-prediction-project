
import streamlit as st
from database import register_user, login_user

def show_auth_page():

    st.title("Time Series Forecasting Platform")
    st.write("Register or log in to continue.")

    login_tab, register_tab = st.tabs(
        ["Login", "Register"]
    )

    with login_tab:
        st.subheader("Welcome Back")

        email = st.text_input(
            "Email Address",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", use_container_width=True):
            if email and password:
                if login_user(email, password):
                    st.session_state["logged_in"] = True
                    st.session_state["user_email"] = email
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
            else:
                st.warning("Please enter email and password.")

    with register_tab:
        st.subheader("Create an Account")

        username = st.text_input(
            "Username",
            key="register_username"
        )

        new_email = st.text_input(
            "Email Address",
            key="register_email"
        )

        new_password = st.text_input(
            "Password",
            type="password",
            key="register_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="confirm_password"
        )

        if st.button("Register", use_container_width=True):
            if not username or not new_email or not new_password:
                st.warning("Please fill in all fields.")

            elif new_password != confirm_password:
                st.error("Passwords do not match.")

            elif len(new_password) < 8:
                st.error("Password must contain at least 8 characters.")

            else:
                success, message = register_user(
                    username,
                    new_email,
                    new_password
                )

                if success:
                    st.success(message)
                else:
                    st.error(message)
