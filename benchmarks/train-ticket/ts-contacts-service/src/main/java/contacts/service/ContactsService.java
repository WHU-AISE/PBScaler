package contacts.service;

import contacts.entity.*;
import edu.fudan.common.util.Response;
import org.springframework.http.HttpHeaders;

import java.util.UUID;

/**
 * @author fdse
 */
public interface ContactsService {

    /**
     * create contacts
     *
     * @param contacts contacts
     * @param headers headers
     * @return Reaponse
     */
    Response createContacts(Contacts contacts, HttpHeaders headers);

    /**
     * create
     *
     * @param addContacts add contacts
     * @param headers headers
     * @return Reaponse
     */
    Response create(Contacts addContacts, HttpHeaders headers);

    /**
     * delete
     *
     * @param contactsId contacts id
     * @param headers headers
     * @return Reaponse
     */
    Response delete(UUID contactsId, HttpHeaders headers);

    /**
     * modify
     *
     * @param contacts contacts
     * @param headers headers
     * @return Reaponse
     */
    Response modify(Contacts contacts, HttpHeaders headers);

    /**
     * get all contacts
     *
     * @param headers headers
     * @return Reaponse
     */
    Response getAllContacts(HttpHeaders headers);

    /**
     * find contacts by id
     *
     * @param id id
     * @param headers headers
     * @return Reaponse
     */
    Response findContactsById(UUID id, HttpHeaders headers);

    /**
     * find contacts by account id
     *
     * @param accountId account id
     * @param headers headers
     * @return Reaponse
     */
    Response findContactsByAccountId(UUID accountId, HttpHeaders headers);

}
