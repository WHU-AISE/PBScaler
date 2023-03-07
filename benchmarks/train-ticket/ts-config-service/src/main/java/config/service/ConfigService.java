package config.service;

import config.entity.Config;
import edu.fudan.common.util.Response;
import org.springframework.http.HttpHeaders;


/**
 * @author fdse
 */
public interface ConfigService {

    /**
     * create by config information and headers
     *
     * @param info info
     * @param headers headers
     * @return Response
     */
    Response create(Config info, HttpHeaders headers);

    /**
     * update by config information and headers
     *
     * @param info info
     * @param headers headers
     * @return Response
     */
    Response update(Config info, HttpHeaders headers);

    /**
     * Config retrieve
     *
     * @param name name
     * @param headers headers
     * @return Response
     */
    Response query(String name, HttpHeaders headers);

    /**
     * delete by name and headers
     *
     * @param name name
     * @param headers headers
     * @return Response
     */
    Response delete(String name, HttpHeaders headers);

    /**
     * query all by headers
     *
     * @param headers headers
     * @return Response
     */
    Response queryAll(HttpHeaders headers);
}
